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


MORTAR_URL = "https://beta-api.mortardata.org"

# connect to Mortar client
client = pymortar.Client(MORTAR_URL)

# initialize container for query information
query = dict()

# airflow_query = """SELECT ?airflow ?vav WHERE { 
#     ?airflow    rdf:type                    brick:Supply_Air_Flow_Sensor .
#     ?vav        rdf:type                    brick:VAV .
#     ?vav        brick:hasPoint              ?airflow .
# }"""

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


################
################
### Fetch
################
################


avail_sites = ['hart', 'gha_ics']
airflow_sensors = client.data_sparql(airflow_query, sites=avail_sites, memsize=4e9)

airflow_view = client.sparql(airflow_query, sites=avail_sites)
airflow_view = airflow_view.reset_index(drop=True)

# select_sites = airflow_view["airflow"].str.contains("|".join(avail_sites).upper())
# af_select_sites = airflow_view.loc[select_sites, :].sort_values(by=["airflow"]).reset_index(drop=True)


# data_found = []
# for i, row in airflow_view.iterrows():
#     df = client.data_uris(row["airflow"])

#     if df._data.empty:
#         data_found.append(False)
#     else:
#         data_found.append(True)


################
################
### Clean Data
################
################

# sort by id and time
airflow_sensors._data = airflow_sensors._data.sort_values(["id", "time"])


################
################
### Retreive Brick Model
################
################
# # get building brick graphs
# hart_model_file = './brick_models/hart_model.ttl'
# gha_model_file = './brick_models/gha_model.ttl'

# g_hart = client.get_graph('hart')
# g_gha = client.get_graph('gha_ics')

# g = brickschema.Graph(brick_version="1.2")
# g_gha_brick = g.parse(g_gha, format="ttl")

# g = brickschema.Graph(brick_version="1.2")
# g_hart_brick = g.parse(io.BytesIO(g_hart), format="turtle")

# # save graphs
# with open(hart_model_file, 'wb') as f:
#     f.write(g_hart)

# with open(gha_model_file, 'wb') as f:
#     f.write(g_gha)


# g = brickschema.Graph(brick_version="1.2")
# g_hart_model = g.load_file(hart_model_file)

# g = brickschema.Graph(brick_version="1.2")
# g_gha_model = g.load_file(gha_model_file, format="ttl")

# get building details
hart_vav2room_file = './bldg_details/hart vav to room schedule.csv'
hart_vav_design_file = './bldg_details/hart vav schedule.csv'

hart_vav2room = pd.read_csv(hart_vav2room_file)
hart_vav_design = pd.read_csv(hart_vav_design_file)

vav_details = {
    "hart": {
        "vav2room": hart_vav2room,
        "vav_design": hart_vav_design,
        },
    }

################
################
### Evaluate Zone Airflow
################
################

def find_min_max_design_airflow(sensor, airflow_view, vav_details):
    """
    Retrieve minimum and maximum design airflow rates from vav box
    """

    sensor_metadata = airflow_view.loc[airflow_view["airflow"].str.contains(sensor), :]

    if sensor_metadata.empty:
        return None, None

    if sensor_metadata.shape[0] > 1:
        import pdb; pdb.set_trace()

    site = sensor_metadata.iloc[0]["site"]
    zone = sensor_metadata.iloc[0]["zone"]
    vav = sensor_metadata.iloc[0]["vav"]

    print(site, "|", vav, "|", zone, "\n")

    # get maximum and minimum cfm
    zone_num = zone.split("#")[1].lower().replace("rm", "")

    room_exists = vav_details[site]["vav2room"]["room"].str.lower().str.contains(zone_num)

    if not any(room_exists):
        return None, None

    room_found = vav_details[site]["vav2room"].loc[room_exists, :]

    if room_found.empty:
        return None, None

    vav_details_exists = vav_details[site]["vav_design"]["vav"].str.contains(room_found.iloc[0]["vav"])
    vav_details_found = vav_details[site]["vav_design"].loc[vav_details_exists, :]

    if vav_details_found.empty:
        return None, None

    vav_max_cfm = vav_details_found.iloc[0]["maximum cfm"]
    vav_min_cfm = vav_details_found.iloc[0]["minimum cfm"]

    return vav_max_cfm, vav_min_cfm


def identify_bldg_occupancy(vav_airflow, occ_hrs=[7,18]):

    mil_time = vav_airflow["time"].dt.hour + (vav_airflow["time"].dt.minute)/60.0

    weekend = vav_airflow["time"].dt.day_name().isin(["Saturday", "Sunday"])
    occupied = (mil_time >= occ_hrs[0]) & (mil_time <= occ_hrs[1]) & ~weekend

    vav_airflow.loc[:, "occupied_hour"] = occupied
    vav_airflow.loc[:, "weekend"] = weekend

    return vav_airflow


def boxplot_params(values, q=[0.25, 0.50, 0.75]):
    """
    Calculate boxplot parameters from the given 
    values pandas data series 
    """

    quants = []
    for q_num in q:
        quants.append(values.quantile(q=q_num))

    iqr = quants[-1] - quants[0]

    upper = quants[-1] + 1.5*iqr
    lower = quants[0] - 1.5*iqr

    # make sure lower and upper are less than data min max values.
    qmax = values.quantile(q=1.00)
    qmin = values.quantile(q=0.00)

    upper = min(upper, qmax)
    lower = max(lower, qmin)

    quants.insert(0, lower)
    quants.append(upper)

    return tuple(quants)


def one_zone_boxplot_set(df, value_col):
    """
    Calculate boxplot parameters for one zone.
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

    plt = figure(x_range=boxplot_cols, height=250, title=plot_name,
            toolbar_location=None)

    upper_design_airflow = Span(location=boxplot_set["maximum_airflow"], dimension="width",
    line_color="gray", line_dash="dashed", line_width=3)
    lower_design_airflow = Span(location=boxplot_set["minimum_airflow"], dimension="width",
    line_color="gray", line_dash="dashed", line_width=3)

    plt.renderers.extend([upper_design_airflow, lower_design_airflow])

    for idx, col in enumerate(boxplot_cols):
        plt = plot_one_boxplot(boxplot_set[col], plt=plt, boxplot_idx=idx+0.5, boxplot_color=boxplot_colors[idx])

    return plt


def plot_boxplots_from_df(boxplot_df):
    pass


# unique_sensors = airflow_view["airflow"].unique()
unique_sensors = airflow_sensors.data["id"].unique()

boxplots = dict()
for sensor in unique_sensors:
    vav_max_cfm, vav_min_cfm = find_min_max_design_airflow(sensor, airflow_view, vav_details)
    vav_airflow = airflow_sensors.data.loc[airflow_sensors.data["id"].isin([sensor]), :]

    vav_airflow = identify_bldg_occupancy(vav_airflow, occ_hrs=[7,18])

    sensor_id = sensor.split("#")[1]
    boxplots[sensor_id] = one_zone_boxplot_set(vav_airflow, "value")

    boxplots[sensor_id].update({
        "maximum_airflow": vav_max_cfm,
        "minimum_airflow": vav_min_cfm
    })

# convert boxplot dict to dataframe
boxplot_df = pd.DataFrame.from_dict(boxplots).transpose()


one_boxplot = boxplot_df.iloc[0]["overall"]
boxplot = plot_one_boxplot(one_boxplot)

output_file(join('./', 'boxplot_test.html'))
save(boxplot)



dev_threshold = 0.25
need_att = []
for idx, boxplot_set in boxplot_df.iterrows():

    plot_name = boxplot_set.name
    p = plot_boxplot_zone_set(boxplot_set, plot_name=plot_name)

    if boxplot_set["minimum_airflow"] is not None:
        airflow_dev = boxplot_set["occupied"] / boxplot_set["minimum_airflow"] - 1
        bad = airflow_dev[2] > dev_threshold
    else:
        airflow_dev = boxplot_set["occupied"] / boxplot_set["unoccupied"][2] - 1
        bad = airflow_dev[2] < dev_threshold

    if bad:
        folder = join("./", "figures", "needs_attention")
        need_att.append(idx)
    else:
        folder  = join("./", "figures", "reasonable")


    filename= join(folder, f"{plot_name}.html")
    output_file(join(filename))
    save(p)



    # plot = plot_airflow_dat(vav_airflow, vav_max_cfm, vav_min_cfm)
