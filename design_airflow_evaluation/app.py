import pymortar
import brickschema
import io
import pandas as pd

from os.path import join

# create plots
from bokeh.palettes import Spectral8, Category20
from bokeh.io import show, save, output_file, export_png
from bokeh.layouts import column
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, RangeTool, LinearAxis, Range1d, BoxAnnotation, Legend, Span


def _query_and_qualify(MORTAR_URL):
    """
    Build query and return sites with VAV airflow sensors

    Parameters
    ----------
    MORTAR_URL: endpoint for mortar database

    Returns
    -------
    query: dictionary containing airflow query
    client: pymortar connection client
    """
    # connect to Mortar client
    client = pymortar.Client(MORTAR_URL)

    # initialize container for query information
    query = dict()

    airflow_query = """SELECT ?airflow ?vav ?zone WHERE {
        ?airflow    a                   brick:Supply_Air_Flow_Sensor .
        ?vav        a                   brick:VAV .
        ?airflow    brick:isPointOf     ?vav .
        ?vav        brick:feeds         ?zone .
        ?zone       a                   brick:HVAC_Zone .
    }"""

    # find sites that qualify for the app
    qualify_resp = client.qualify({"measurement": airflow_query})

    query["query"] = dict()
    query["airflow"] = airflow_query

    print("running on {0} sites".format(len(qualify_resp.sites)))
    print(qualify_resp.sites)

    return query, client


def _fetch(query, client, sites=None):
    """
    Fetch query metadata results and query resulting sensor datastreams

    Parameters
    ----------
    query: dictionary containing airflow query
    client: pymortar connection client

    Returns
    -------
    airflow_view: query metadata results
    airflow_sensors: pymortar Dataset for the sites specified
    """

    airflow_query = query["airflow"]

    airflow_view = client.sparql(airflow_query, sites=sites)
    airflow_sensors = client.data_sparql(airflow_query, sites=sites, memsize=4e9)

    return airflow_view, airflow_sensors


def _clean(airflow_view, airflow_sensors):
    """
    Clean query metadata results and pymortar Dataset

    Parameters
    ----------
    airflow_view: query metadata results
    airflow_sensors: pymortar Dataset for the sites specified

    Returns
    -------
    airflow_view: query metadata results
    airflow_sensors: pymortar Dataset for the sites specified
    """

    # reset index in metadata view
    airflow_view = airflow_view.reset_index(drop=True)

    # sort by id and time
    airflow_sensors._data = airflow_sensors._data.sort_values(["id", "time"])

    return airflow_view, airflow_sensors


def _retreive_bldg_details(sites, vav2room_files, vav_design_files):
    """
    Retrieve vav system details to enable detailed analysis.

    Parameters
    ----------
    sites: site(s) to do detailed analysis
    vav2room_files: path to file(s) that maps room to vav equipment
    vav_design_files: path to file(s) that maps vav to vav design parameters

    Returns
    -------
    vav_details: dictionary with vav details of each specified site
    """

    vav_details = dict()
    for s, vr, vd in zip(sites, vav2room_files, vav_design_files):
        # get building details
        vav2room = pd.read_csv(vr)
        vav_design = pd.read_csv(vd)

        one_site = {
            s: {
                "vav2room": vav2room,
                "vav_design": vav_design,
                },
            }
        vav_details.update(one_site)

    return vav_details


def find_vav_design_details(sensor, airflow_view, vav_details):
    """
    Retrieve minimum and maximum design airflow rates from vav box.

    Parameters
    ----------
    sensor: airflow sensor URI name
    airflow_view: query metadata results
    vav_details: dictionary with vav details of each specified site

    Returns
    -------
    vav_design: dictionary with detailed vav parameters
    """

    # initiate dictionary to hold vav details
    vav_design = dict()

    sensor_metadata = airflow_view.loc[airflow_view["airflow"].str.contains(sensor), :]

    if sensor_metadata.empty:
        return vav_design

    if sensor_metadata.shape[0] > 1:
        import pdb; pdb.set_trace()

    site = sensor_metadata.iloc[0]["site"]
    zone = sensor_metadata.iloc[0]["zone"]
    vav = sensor_metadata.iloc[0]["vav"]

    vav_design[site] = {
        "vav": None,
        "ahu": None,
        "vav_maximum_airflow": None,
        "vav_minimum_airflow": None,
        "ahu_maximum_airflow": None,
        "ahu_minimum_airflow": None,
    }

    print(site, "|", vav, "|", zone, "\n")

    # get maximum and minimum cfm
    zone_num = zone.split("#")[1].lower().replace("rm", "")

    room_exists = vav_details[site]["vav2room"]["room"].str.lower().str.contains(zone_num)

    if not any(room_exists):
        return vav_design

    room_found = vav_details[site]["vav2room"].loc[room_exists, :]

    if room_found.empty:
        return vav_design

    vav_details_exists = vav_details[site]["vav_design"]["vav"].str.contains(room_found.iloc[0]["vav"])
    vav_details_found = vav_details[site]["vav_design"].loc[vav_details_exists, :]

    if vav_details_found.empty:
        return vav_design

    # get values for each design parameter
    for param in vav_design[site].keys():
        param_val = vav_details_found.iloc[0][param]
        vav_design[site].update({param: param_val})

    return vav_design


def identify_bldg_occupancy(vav_airflow, occ_hrs=[7,18]):
    """
    Identify occupancy hours for the datastream downloaded
    
    Parameters
    ----------
    vav_airflow: airflow sensor data stream
    occ_hrs: building occupancy hours [<start of occupancy>, <end of occupancy>]
        in decimal time.

    Returns
    -------
    vav_airflow: airflow sensor data stream with occupied_hour and weekend boolean columns
    """

    mil_time = vav_airflow["time"].dt.hour + (vav_airflow["time"].dt.minute)/60.0

    weekend = vav_airflow["time"].dt.day_name().isin(["Saturday", "Sunday"])
    occupied = (mil_time >= occ_hrs[0]) & (mil_time <= occ_hrs[1]) & ~weekend

    vav_airflow.loc[:, "occupied_hour"] = occupied
    vav_airflow.loc[:, "weekend"] = weekend

    return vav_airflow


def boxplot_params(values, q=[0.25, 0.50, 0.75], whiskers=[0.95, 0.05]):
    """
    Calculate quantiles for defined data e.g. parameters to create
    boxplot.

    Parameters
    ----------
    values: Pandas data series
    q: quantiles to calculate
    whiskers: upper and lower quantile for boxplot whiskers


    Returns
    -------
    quants: tuple for the calculated quantiles
    """

    quants = []
    for q_num in q:
        quants.append(values.quantile(q=q_num))

    iqr = quants[-1] - quants[0]

    upper = quants[-1] + 1.5*iqr
    lower = quants[0] - 1.5*iqr

    # make sure lower and upper are less than data min max values.
    qmax = values.quantile(q=whiskers[0])
    qmin = values.quantile(q=whiskers[1])

    upper = min(upper, qmax)
    lower = max(lower, qmin)

    quants.insert(0, lower)
    quants.append(upper)

    return tuple(quants)


def one_zone_boxplot_set(df, value_col):
    """
    Calculate boxplot parameters for one zone.

    Parameters
    ----------
    df: pandas dataframe with data
    value_col: name of column containing the values to analyze

    Returns
    -------
    boxplots: dictionary of boxplot parameters for different
        categories defined below.

        Boxplot 1: all data
        Boxplot 2: occupancy hours
        Boxplot 3: unoccupied hours including all weekend
        Boxplot 4: Only weekends
    """

    overall = boxplot_params(df.loc[:, value_col])
    occupied = boxplot_params(df.loc[df["occupied_hour"], value_col])
    unoccupied = boxplot_params(df.loc[~df["occupied_hour"], value_col])
    weekend = boxplot_params(df.loc[df["weekend"], value_col])

    boxplots = {
        "overall": overall,
        "occupied": occupied,
        "unoccupied": unoccupied,
        "weekend": weekend,
    }

    return boxplots


def calculate_quartiles(airflow_sensors, airflow_view, vav_details, occ_hrs=[7,18]):
    """
    Calculate quartiles for each vav airflow sensor measurements

    Parameters
    ----------
    airflow_sensors: pymortar Dataset for the sites specified
    airflow_view: query metadata results
    vav_details: dictionary with vav details of each specified site
    occ_hrs: building occupancy hours [<start of occupancy>, <end of occupancy>]
        in decimal time.

    Returns
    -------
    boxplot_df: pandas dataframe containing calculated quartiles for each vav airflow sensor
    """

    # unique_sensors = airflow_view["airflow"].unique()
    unique_sensors = airflow_sensors.data["id"].unique()

    boxplots = dict()
    for sensor in unique_sensors:
        vav_design = find_vav_design_details(sensor, airflow_view, vav_details)
        vav_airflow = airflow_sensors.data.loc[airflow_sensors.data["id"].isin([sensor]), :]

        vav_airflow = identify_bldg_occupancy(vav_airflow, occ_hrs=occ_hrs)

        sensor_id = sensor.split("#")[1]
        boxplots[sensor_id] = one_zone_boxplot_set(vav_airflow, "value")

        if bool(vav_design):
            site = list(vav_design.keys())[0]
            boxplots[sensor_id].update({"site": site})
            boxplots[sensor_id].update(vav_design[site])

    # convert boxplot dict to dataframe
    boxplot_df = pd.DataFrame.from_dict(boxplots).transpose()

    return boxplot_df


def plot_airflow_dat(vav_airflow, vav_max_cfm, vav_min_cfm):

    dat_src = ColumnDataSource(vav_airflow)
    x_range_str_time = vav_airflow.iloc[0]["time"]
    x_range_end_time = vav_airflow.iloc[-1]["time"]

    p = figure(
        plot_height=300, plot_width=1500,
        x_axis_type="datetime", x_axis_location="below",
        x_range=(x_range_str_time, x_range_end_time)
    )
    p.add_layout(Legend(), 'right')
    p.yaxis.axis_label = "VAV Airflow Rate [CFM]"

    p.step('time', 'value', source=dat_src, line_width=2)

    plot_name = '{}-airflow_timeseries.html'.format(vav_airflow.iloc[0]["id"].split("#")[1])
    output_file(join('./', plot_name))
    save(p)

    return p


def plot_one_boxplot(one_boxplot_params, plt=None, boxplot_idx=None, boxplot_color="#0080ff"):

    if plt is None:
        ini_plot = figure(tools="", toolbar_location=None)
        boxplot_idx = 1

    box_width = 0.5

    # plot whiskers
    plt.segment(boxplot_idx, one_boxplot_params[-1], boxplot_idx, one_boxplot_params[3], line_width=2, line_color="black")
    plt.segment(boxplot_idx, one_boxplot_params[0], boxplot_idx, one_boxplot_params[1], line_width=2, line_color="black")

    # plot boxes
    plt.vbar(boxplot_idx, box_width, one_boxplot_params[2], one_boxplot_params[3], fill_color=boxplot_color, line_color="black")
    plt.vbar(boxplot_idx, box_width, one_boxplot_params[1], one_boxplot_params[2], fill_color=boxplot_color, line_color="black")

    # plot median bar
    plt.rect(boxplot_idx, one_boxplot_params[2], box_width, 0.02, line_width=3, line_color="black")

    # # plot whisker endpoints
    # plt.rect(boxplot_idx, one_boxplot_params[0], 0.2, 0.01, line_color="black")
    # plt.rect(boxplot_idx, one_boxplot_params[-1], 0.2, 0.01, line_color="black")

    plt.xgrid.grid_line_color = None
    plt.ygrid.grid_line_color = "white"
    plt.grid.grid_line_width = 2
    plt.xaxis.major_label_text_font_size="16px"

    return plt


def plot_boxplot_zone_set(boxplot_set, plot_name):

    boxplot_cols = ["overall", "occupied", "unoccupied", "weekend"]
    boxplot_colors = ["#4da6ff", "#4dffa6", "#ff4d4d", "#ffa64d"]

    plt = figure(
        x_range=boxplot_cols, height=250, title=plot_name,
        toolbar_location=None
        )

    max_airflow = boxplot_set["vav_maximum_airflow"]
    min_airflow = boxplot_set["vav_minimum_airflow"]

    upper_design_airflow = Span(location=max_airflow, dimension="width",
    line_color="gray", line_dash="dashed", line_width=3)
    lower_design_airflow = Span(location=min_airflow, dimension="width",
    line_color="gray", line_dash="dashed", line_width=3)

    plt.renderers.extend([upper_design_airflow, lower_design_airflow])

    max_y = max_airflow
    min_y = min_airflow
    for idx, col in enumerate(boxplot_cols):
        boxplot_params = boxplot_set[col]
        plt = plot_one_boxplot(boxplot_params, plt=plt, boxplot_idx=idx+0.5, boxplot_color=boxplot_colors[idx])

        if min_airflow is not None:
            max_y = max(max_y, max(boxplot_params))
            min_y = min(min_y, min(boxplot_params))

    if min_airflow is not None:
        plt.y_range = Range1d(min_y, max_y)

    return plt


def calculate_min_airflow_deviation(boxplot_df):
    """
    Calculate the percentage deviation from minimum design airflow rate.

    Parameters
    ----------
    boxplot_df: pandas dataframe containing calculated quartiles for each vav airflow sensor

    Results
    ----------
    """

    # define holding column names
    pct_dev_col = ["median_pct_dev_from_dmin_occ", "median_pct_dev_from_dmin_unocc"]
    abs_dev_col = ["median_abs_dev_form_dmin_occ", "median_abs_dev_form_dmin_unocc"]
    timeframes = ["occupied", "unoccupied"]

    # initialize holding columns
    for kdx, tf in enumerate(timeframes):
        boxplot_df.loc[:, abs_dev_col[kdx]] = None
        boxplot_df.loc[:, pct_dev_col[kdx]] = None

    for idx, sensor in boxplot_df.iterrows():
        min_flow = sensor["vav_minimum_airflow"]
        if min_flow is not None:
            for jdx, tf in enumerate(timeframes):
                abs_airflow_dev = sensor[tf] - min_flow
                airflow_dev = sensor[tf] / min_flow - 1

                # insert calculation in dataframe
                boxplot_df.loc[idx, abs_dev_col[jdx]] = abs_airflow_dev[2]
                boxplot_df.loc[idx, pct_dev_col[jdx]] = airflow_dev[2]

    return boxplot_df


def _analyze(airflow_sensors, airflow_view, vav_details, occ_hrs=[6,22], dev_threshold=0.15):
    """
    Analyze each vav airflow sensor
    """
    boxplot_df = calculate_quartiles(airflow_sensors, airflow_view, vav_details, occ_hrs)
    boxplot_df = calculate_min_airflow_deviation(boxplot_df)

    # save spreadsheet
    boxplot_df.sort_values(["median_pct_dev_from_dmin_occ"]).to_csv("summary_of_vav_airflow_analysis.csv")

    need_att = []
    for idx, sensor in boxplot_df.iterrows():

        plot_name = f"{sensor.name} | Bldg. Occ = {occ_hrs}"
        p = plot_boxplot_zone_set(sensor, plot_name=plot_name)

        airflow_dev = sensor["median_pct_dev_from_dmin_occ"]
        if airflow_dev is not None:
            if airflow_dev > dev_threshold:
                subfolder = 'too high'
                bad = True

            elif airflow_dev < -1*dev_threshold:
                subfolder = 'too low'
                bad = True

            else:
                bad = False

        if bad:
            folder = join("./", "figures", "needs_attention", subfolder)
            need_att.append(idx)
        else:
            folder  = join("./", "figures", "reasonable")

        filename= join(folder, f"{plot_name}.html")
        output_file(join(filename))
        save(p)


if __name__ == "__main__":

    ################
    ################
    ### Query and qualify
    ################
    ################

    MORTAR_URL = "https://beta-api.mortardata.org"
    query, client = _query_and_qualify(MORTAR_URL)

    ################
    ################
    ### Fetch and clean
    ################
    ################

    avail_sites = ['hart', 'gha_ics']
    airflow_view, airflow_sensors = _fetch(query, client, sites=avail_sites)
    airflow_view, airflow_sensors = _clean(airflow_view, airflow_sensors)

    ################
    ################
    ### Retreive detailed building vav system information
    ################
    ################

    hart_vav2room_file = './bldg_details/hart vav to room schedule.csv'
    hart_vav_design_file = './bldg_details/hart vav schedule.csv'

    vav_sites = [avail_sites[0]]
    vav2room_files = [hart_vav2room_file]
    vav_design_files = [hart_vav_design_file]

    vav_details = _retreive_bldg_details(vav_sites, vav2room_files, vav_design_files)

    ################
    ################
    ### Analysis: Evaluate Zone Airflow
    ################
    ################

    _analyze(airflow_sensors, airflow_view, vav_details, occ_hrs=[6,22], dev_threshold=0.15)


        # plot = plot_airflow_dat(vav_airflow, vav_max_cfm, vav_min_cfm)
