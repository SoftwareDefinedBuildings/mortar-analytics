import brickschema
import pandas as pd
import numpy as np
from os.path import join
import sys
sys.path.append("/mnt/c/Users/duar3/Documents/github/smap/python")
sys.path.append("/mnt/c/Users/duar3/Documents/github/smap/python/smap")

from smap.archiver.client import SmapClient
from smap.contrib import dtutil

# create plots
from bokeh.palettes import Spectral8, Category20
from bokeh.io import show, save, output_file
from bokeh.layouts import column
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, RangeTool, LinearAxis, Range1d, BoxAnnotation, Legend

_debug = 1

def _query_hw_consumers(g):
    """
    Retrieve hot water consumers in the building, their respective
    boiler(s), and relevant hvac zones.
    """
    # query direct and indirect hot water consumers
    hw_consumers_query = """SELECT DISTINCT * WHERE {
    ?boiler     rdf:type/rdfs:subClassOf?   brick:Boiler .
    ?boiler     brick:feeds+                ?t_unit .
    ?t_unit     rdf:type                    ?equip_type .
    ?mid_equip  brick:feeds                 ?t_unit .
    ?t_unit     brick:feeds+                ?room_space .
    ?room_space rdf:type/rdfs:subClassOf?   brick:HVAC_Zone .

        FILTER NOT EXISTS {
            ?subtype ^a ?t_unit ;
                (rdfs:subClassOf|^owl:equivalentClass)* ?equip_type .
            filter ( ?subtype != ?equip_type )
            }
    }
    """
    if _debug: print("Retrieving hot water consumers for each boiler.\n")

    q_result = g.query(hw_consumers_query)
    df_hw_consumers = pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars])

    return df_hw_consumers


def _clean_metadata(df_hw_consumers):
    """
    Cleans metadata dataframe to have unique hot water consumers with
    most specific classes associated to other relevant information.
    """

    unique_t_units = df_hw_consumers.loc[:, "t_unit"].unique()
    direct_consumers_bool = df_hw_consumers.loc[:, 'mid_equip'] == df_hw_consumers.loc[:, 'boiler']

    direct_consumers = df_hw_consumers.loc[direct_consumers_bool, :]
    indirect_consumers = df_hw_consumers.loc[~direct_consumers_bool, :]

    # remove any direct hot consumers listed in indirect consumers
    for unit in direct_consumers.loc[:, "t_unit"].unique():
        indir_test = indirect_consumers.loc[:, "t_unit"] == unit

        # update indirect consumers df
        indirect_consumers = indirect_consumers.loc[~indir_test, :]

    # label type of hot water consumer
    direct_consumers.loc[:, "consumer_type"] = "direct"
    indirect_consumers.loc[:, "consumer_type"] = "indirect"

    hw_consumers = pd.concat([direct_consumers, indirect_consumers])
    hw_consumers = hw_consumers.drop(columns=["subtype"]).reset_index(drop=True)

    return hw_consumers


def search_for_entities(g, class_type, point_list, relationship="brick:hasPoint"):
    """
    Return entities with the defined class type
    """

    if isinstance(point_list, list):
        points = " ".join(point_list)

    type_query = f"""SELECT DISTINCT * WHERE {{
        VALUES          ?req_point {{ {points} }}
        ?entity         rdf:type/rdfs:subClassOf?   {class_type} .
        ?entity         {relationship}              ?entity_points .
        ?entity_points  rdf:type/rdfs:subClassOf?   ?req_point .
        ?entity         brick:isPartOf?             ?larger_comp .
        ?larger_comp    rdf:type                    ?larger_comp_class .

        FILTER NOT EXISTS {{
            ?subtype ^a ?larger_comp ;
                (rdfs:subClassOf|^owl:equivalentClass)* ?larger_comp_class .
            filter ( ?subtype != ?larger_comp_class )
            }}
    }}
    """

    q_result = g.query(type_query)

    df = pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars])
    #df = df.drop_duplicates(subset=['point_name']).reset_index(drop=True)

    return df



def return_entity_points(g, entity, point_list):
    """
    Return defined brick point class for piece of equipment
    """
    
    if isinstance(point_list, list):
        points = " ".join(point_list)

    # query to return certain points of other points
    term_query = f"""SELECT DISTINCT * WHERE {{
        VALUES ?req_point {{ {points} }}
        ?point_name     rdf:type                        ?req_point .
        ?point_name     brick:isPointOf                 ?t_unit .
        ?point_name     brick:bacnetPoint               ?bacnet_id .
        ?point_name     brick:hasUnit?                  ?val_unit .
        ?bacnet_id      brick:hasBacnetDeviceInstance   ?bacnet_instance .
        ?bacnet_id      brick:hasBacnetDeviceType       ?bacnet_type .
        ?bacnet_id      brick:accessedAt                ?bacnet_net .
        ?bacnet_net     dbc:connstring                  ?bacnet_addr .
        }}"""

    # execute the query
    q_result = g.query(term_query, initBindings={"t_unit": entity})

    df = pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars])
    df = df.drop_duplicates(subset=['point_name']).reset_index(drop=True)

    return df


def get_paths_from_tags(tags):
    paths = {key: tags[key]["Path"] for key in tags}
    paths = pd.DataFrame.from_dict(paths, orient='index', columns=['path'])
    new_cols = ["empty", "site", "bms", "bacnet_instance", "bms2", "point_name"]

    # adjustments to dataframe
    paths[new_cols] = paths.path.str.split("/", expand=True)
    paths = paths.drop(columns=["empty"])

    return paths


def plot_multiple_entities(metadata, data, start, end, filename, exclude_str=None, ylimits=None):

    plots = []
    for ii, point_type in enumerate(metadata['req_point'].unique()):
        # if "Position" in point_type:
        #     y_plot_range = Range1d(start=0, end=101)
        # else:
        #     y_plot_range = Range1d(start=0, end=1.1)

        # plot settings
        plt_colors = Category20[20]

        x_range_str_time = pd.to_datetime(start, unit='s', utc=True).tz_convert('US/Pacific').tz_localize(None)
        x_range_end_time = pd.to_datetime(end, unit='s', utc=True).tz_convert('US/Pacific').tz_localize(None)

        if ii == 0:
            x_plot_range = (x_range_str_time, x_range_end_time)
        else:
            x_plot_range = plots[0].x_range

        p = figure(
            plot_height=300, plot_width=1500,
            x_axis_type="datetime", x_axis_location="below",
            x_range=x_plot_range,
            # y_range=y_plot_range
            )
        p.add_layout(Legend(), 'right')

        in_data = metadata["req_point"].isin([point_type])
        in_data_index = in_data[in_data].index
        df_subset = [data[x] for x in in_data_index]

        for i, dd in enumerate(df_subset):
            if exclude_str is not None:
                if any([nm in metadata.loc[in_data_index[i], "point_name_x"] for nm in exclude_str]):
                    continue
            p.step(
                pd.to_datetime(dd[:, 0], unit='ms', utc=True).tz_convert("US/Pacific").tz_localize(None),
                dd[:, 1], legend_label=metadata.loc[in_data_index[i], "point_name_x"],
                color = plt_colors[i % len(plt_colors)], line_width=2,
                mode = 'after'
                )

        y_axis_label = str(point_type).split("#")[1]
        p.yaxis.axis_label = y_axis_label

        if ylimits is not None:
            y_plot_range = Range1d(start=ylimits[0], end=ylimits[1])
            p.y_range = y_plot_range

        p.legend.click_policy = "hide"

        p.legend.label_text_font_size = "9px"
        p.legend.label_height = 5
        p.legend.glyph_height = 5
        p.legend.spacing = 5

        plots.append(p)

    output_file(filename)
    save(column(plots))

    return plots


def read_requests_data(hwc_request_file):

    # read and setup hot water consumer file that contains
    # number of requests from fast and slow reacting, and Total Requests Sent
    # to controller
    hwcr_dat = pd.read_csv(hwc_request_file, header=None, index_col=3, parse_dates=True)
    hwcr_dat.index.name = "timestamp"
    hwcr_dat.columns = ["Fast Response Requests", "HTM Requests", "Total Requests Sent"]

    return hwcr_dat

def read_new_sp_data(boiler_sp_file):
    # read and setup boiler setpoint file that contains
    # new hot water setpoint calculated by controller
    bsp_dat = pd.read_csv(boiler_sp_file, header=None, index_col=0, parse_dates=True)
    bsp_dat.index.name = "timestamp"
    bsp_dat.columns = ["Controller New HW Setpoint"]

    # convert to degF
    bsp_dat["Controller New HW Setpoint"] = (bsp_dat["Controller New HW Setpoint"] - 273.15)*1.8 + 32

    return bsp_dat

def add_ctrl_data(plt, boiler_sp_file):

    bsp_dat = read_new_sp_data(boiler_sp_file)

    # add data to plot
    col_plt = "Controller New HW Setpoint"
    plt.step(
            bsp_dat.index,
            bsp_dat[col_plt], legend_label=col_plt,
            color = "#d53e4f", line_width=2.5, line_dash="dashed",
            mode = 'after'
            )

    return plt

def add_req_num_data(plt, hwc_request_file):

    hwcr_dat = read_requests_data(hwc_request_file)

    # add data to plot
    col_plt = ["Total Requests Sent", "HTM Requests"]
    colors = ["#4fd53e", "#d5c43e"]

    new_p = figure(
        plot_height=150, plot_width=1500,
        x_axis_type="datetime", x_axis_location="below",
        x_range=plt.x_range,
        y_range=Range1d(start=0, end=15)
        )
    new_p.add_layout(Legend(), 'right')

    for dd, col in enumerate(col_plt):
        new_p.step(
            hwcr_dat.index,
            hwcr_dat[col], legend_label=col,
            color = colors[dd], line_width=2,
            mode = 'after'
            )

    return new_p


def plot_boiler_temps(boiler_points_to_download, boiler_data, filename, ctrlr_sp=None, req_num=None):

    # plot settings
    plt_colors = Spectral8

    x_range_str_time = pd.to_datetime(start, unit='s', utc=True).tz_convert('US/Pacific').tz_localize(None)
    x_range_end_time = pd.to_datetime(end, unit='s', utc=True).tz_convert('US/Pacific').tz_localize(None)

    p = figure(
            plot_height=300, plot_width=1500,
            x_axis_type="datetime", x_axis_location="below",
            x_range=(x_range_str_time, x_range_end_time),
            y_range=Range1d(start=0, end=200)
            )
    p.add_layout(Legend(), 'right')
    p.yaxis.axis_label = "Boiler temperatures"

    for i, dd in enumerate(boiler_data):
        p.step(
            pd.to_datetime(dd[:, 0], unit='ms', utc=True).tz_convert("US/Pacific").tz_localize(None),
            dd[:, 1], legend_label=boiler_points_to_download.iloc[i]["point_name_x"],
            color = plt_colors[i % len(plt_colors)], line_width=2,
            mode = 'after'
            )

    # add extra plot lines
    if ctrlr_sp is not None:
        p = add_ctrl_data(p, ctrlr_sp)

    if req_num is not None:
        new_p = add_req_num_data(p, req_num)
        new_p.yaxis.axis_label = "HW Requests"

        plots = [p, new_p]
    else:
        plots = [p]

    for plt in plots:
        plt.legend.click_policy = "hide"

        plt.legend.label_text_font_size = "9px"
        plt.legend.label_height = 5
        plt.legend.glyph_height = 5
        plt.legend.spacing = 5

    output_file(filename)
    save(column(plots))

    return plots


def get_data_from_smap(points_to_download, paths, smap_client, start, end):
    data_ids = points_to_download["bacnet_instance"]
    avail_to_download = paths["bacnet_instance"].isin(data_ids)
    data_paths = paths.loc[avail_to_download, :]

    # combine the data frames
    df_combine = pd.merge(data_paths.reset_index(), points_to_download, how="right", on="bacnet_instance")

    # get data from smap
    data = smap_client.data_uuid(df_combine["index"], start, end, cache=False)

    return df_combine, data


def convert_smap_to_pandas(smap_dat_arr, col_labels=None):
    """
    Convert a dataset downloaded from smap to a pandas dataframe
    """

    df = []
    for i, dd in enumerate(smap_dat_arr):
        # df_timestamps = pd.to_datetime(dd[:, 0], unit='ms', utc=True).tz_convert("US/Pacific").tz_localize(None) # does not take into account daylight savings time change
        df_timestamps = pd.to_datetime(dd[:, 0], unit='ms', utc=True).tz_convert("US/Pacific")
        if col_labels is not None:
            cur_df = pd.DataFrame(dd[:, 1], index=df_timestamps, columns=[col_labels[i]])
        else:
            cur_df = pd.DataFrame(dd[:, 1], index=df_timestamps)

        # add cur_df to container
        df.append(cur_df)

    # combine all individual timeseries
    all_dfs = pd.concat(df, axis=1)

    return all_dfs


if __name__ == "__main__":
    # database settings
    url = "http://178.128.64.40:8079"
    keyStr = "B7qm4nnyPVZXbSfXo14sBZ5laV7YY5vjO19G"
    where = "Metadata/SourceName = 'Field Study 4'"

    # set file names
    exp_brick_model_file = "../dbc_brick_expanded.ttl"
    boiler_sp_file = join("DATA", "boiler_setpoint.csv")
    hwc_request_file = join("DATA", "number_of_request.csv")

    # set save folder names
    plot_folder = "./figures"

    # time interval for to download data
    start = dtutil.dt2ts(dtutil.strptime_tz("01-01-2023", "%m-%d-%Y"))
    end   = dtutil.dt2ts(dtutil.strptime_tz("04-01-2023", "%m-%d-%Y"))

    # initiate smap client and download tags
    smap_client = SmapClient(url, key=keyStr)
    tags = smap_client.tags(where, asdict=True)

    # retrieve relevant tags from smap database
    paths = get_paths_from_tags(tags)

    # load schema files
    g = brickschema.Graph()
    g.load_file(exp_brick_model_file)

    # query hot water consumers and clean metadata
    df_hw_consumers = _query_hw_consumers(g)
    df_hw_consumers = _clean_metadata(df_hw_consumers)


    #############################
    ##### Return hw consumer ctrl points
    #############################
    vlvs = ["brick:Position_Sensor", "brick:Valve_Command"]
    df_vlvs = []
    for t_unit in df_hw_consumers["t_unit"].unique():
        df_vlvs.append(return_entity_points(g, t_unit, vlvs))

    df_vlvs = pd.concat(df_vlvs).reset_index(drop=True)
    df_vlvs["bacnet_instance"] = df_vlvs["bacnet_instance"].astype(int).astype(str)

    # download data from smap
    ctrl_points_to_download, hw_ctrl_data = get_data_from_smap(df_vlvs, paths, smap_client, start, end)


    # create plot
    fig_file = join(plot_folder, "hw_consumer_ctrl.html")
    ctrl_plots = plot_multiple_entities(ctrl_points_to_download, hw_ctrl_data, start, end, fig_file, exclude_str=["REV", "DPR", "D-O"])

    #############################
    ##### Return boiler points
    #############################
    temps = [
        "brick:Hot_Water_Supply_Temperature_Sensor",
        "brick:Return_Water_Temperature_Sensor",
        "brick:Supply_Water_Temperature_Setpoint"
        ]
    df_hw_temps = []
    for boiler in df_hw_consumers["boiler"].unique():
        df_hw_temps.append(return_entity_points(g, boiler, temps))

    df_hw_temps = pd.concat(df_hw_temps).reset_index(drop=True)
    df_hw_temps["bacnet_instance"] = df_hw_temps["bacnet_instance"].astype(int).astype(str)

    # download data from smap
    boiler_points_to_download, boiler_data = get_data_from_smap(df_hw_temps, paths, smap_client, start, end)

    # create plots
    fig_file = join(plot_folder, "boiler_temps.html")
    boiler_plot = plot_boiler_temps(boiler_points_to_download, boiler_data, fig_file, ctrlr_sp=boiler_sp_file, req_num=hwc_request_file)

    # save data streams for later processing
    boiler_df = convert_smap_to_pandas(boiler_data, col_labels=boiler_points_to_download["point_name_x"])
    boiler_df.to_csv(join('./', 'DATA', 'smap_boiler_temps.csv'), index_label='Timestamp')

    #############################
    ##### Return hw consumer discharge temperatures
    #############################

    dischrg_temps = ["brick:Supply_Air_Temperature_Sensor", "brick:Embedded_Temperature_Sensor"]

    df_dischrg_temps = []
    for t_unit in df_hw_consumers["t_unit"].unique():
        df_dischrg_temps.append(return_entity_points(g, t_unit, dischrg_temps))

    df_dischrg_temps = pd.concat(df_dischrg_temps).reset_index(drop=True)
    df_dischrg_temps["bacnet_instance"] = df_dischrg_temps["bacnet_instance"].astype(int).astype(str)

    # download data from smap
    # TODO: there is a value error when cache is set to true
    dischrg_temps_to_download, dischrg_temps_data = get_data_from_smap(df_dischrg_temps, paths, smap_client, start, end)

    # create plots
    fig_file = join(plot_folder, "hw_consumer_discharge_temps.html")
    dischrg_temps_plots = plot_multiple_entities(dischrg_temps_to_download, dischrg_temps_data, start, end, fig_file)


    #############################
    ##### Return zone temperatures
    #############################

    zone_temps = ["brick:Zone_Air_Temperature_Sensor", "brick:Air_Temperature_Setpoint"]

    df_zone_temps = []
    for zn in df_hw_consumers["room_space"]:
        df_zone_temps.append(return_entity_points(g, zn, zone_temps))

    df_zone_temps = pd.concat(df_zone_temps).reset_index(drop=True)
    df_zone_temps["bacnet_instance"] = df_zone_temps["bacnet_instance"].astype(int).astype(str)

    # download data from smap
    # TODO: there is a value error when cache is set to true
    zn_temps_to_download, zn_temps_data = get_data_from_smap(df_zone_temps, paths, smap_client, start, end)

    # create plots
    air_zones = zn_temps_to_download["t_unit"].str.contains("Air_Zone")
    rad_zones = zn_temps_to_download["t_unit"].str.contains("Radiant_Zone")

    fig_file_air = join(plot_folder, "air_zone_temps.html")
    fig_file_rad = join(plot_folder, "rad_zone_temps.html")
    air_zone_temps_plots = plot_multiple_entities(zn_temps_to_download.loc[air_zones, :], zn_temps_data, start, end, fig_file_air)
    rad_zone_temps_plots = plot_multiple_entities(zn_temps_to_download.loc[rad_zones, :], zn_temps_data, start, end, fig_file_rad, ylimits=(55,85))


    #############################
    ##### Return pump speed status
    #############################
    pumps = ["brick:Speed_Status", "brick:On_Off_Status"]
    pumps_metadata = search_for_entities(g, "brick:Pump", pumps, relationship="brick:hasPoint")

    df_pmp = []
    for t_unit in pumps_metadata["entity"].unique():
        df_pmp.append(return_entity_points(g, t_unit, pumps))

    df_pmp = pd.concat(df_pmp).reset_index(drop=True)
    df_pmp["bacnet_instance"] = df_pmp["bacnet_instance"].astype(int).astype(str)

    # download data from smap
    do_not_include = df_pmp["point_name"].str.contains("Analog")
    pump_points_to_download, pump_data = get_data_from_smap(df_pmp.loc[~do_not_include, :], paths, smap_client, start, end)

    # create plot
    fig_file = join(plot_folder, "hydronic_plant_pumps.html")
    pump_status_plots = plot_multiple_entities(pump_points_to_download, pump_data, start, end, fig_file, exclude_str=["Analog"])

    #############################
    ##### AHU discharge temps
    #############################
    ahu_dchrg = ["brick:Discharge_Air_Temperature_Sensor"]
    ahu_metadata = search_for_entities(g, "brick:AHU", ahu_dchrg, relationship="brick:hasPoint")

    df_ahu = []
    for t_unit in ahu_metadata["entity"].unique():
        df_ahu.append(return_entity_points(g, t_unit, ahu_dchrg))

    df_ahu = pd.concat(df_ahu).reset_index(drop=True)
    df_ahu["bacnet_instance"] = df_ahu["bacnet_instance"].astype(int).astype(str)

    ahu_points_to_download, ahu_data = get_data_from_smap(df_ahu, paths, smap_client, start, end)

    # create plots
    fig_file = join(plot_folder, "ahu_discharge_temps.html")
    dischrg_temps_plots = plot_multiple_entities(ahu_points_to_download, ahu_data, start, end, fig_file)

    #############################
    ##### Heat pump heating start and stop status
    #############################
    hp_status = ["brick:Heating_Start_Stop_Status", "brick:On_Off_Command"]
    hp_metadata = search_for_entities(g, "brick:Water_Source_Heat_Pump", hp_status, relationship="brick:hasPoint")

    df_asset = []
    for ind_asset in hp_metadata["entity"].unique():
        df_asset.append(return_entity_points(g, ind_asset, hp_status))

    df_asset = pd.concat(df_asset).reset_index(drop=True)
    df_asset["bacnet_instance"] = df_asset["bacnet_instance"].astype(int).astype(str)

    # filter
    hp_sf_bool = df_asset["point_name"].str.contains("SF-C")
    hp_heat_status_bool = df_asset["req_point"].str.contains("Heating_Start_Stop_Status")
    filter_bool = np.logical_or(hp_sf_bool, hp_heat_status_bool)
    df_asset = df_asset.loc[filter_bool, :]

    hp_points_to_download, hp_data = get_data_from_smap(df_asset, paths, smap_client, start, end)

    # create plots
    fig_file = join(plot_folder, "hp_heating_status.html")
    hp_status_plots = plot_multiple_entities(hp_points_to_download, hp_data, start, end, fig_file)
