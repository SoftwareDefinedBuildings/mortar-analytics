import pymortar
import brickschema
import io
import pandas as pd

from os.path import join

# create plots
from bokeh.palettes import Spectral8, Category20
from bokeh.io import show, save, output_file
from bokeh.layouts import column
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, RangeTool, LinearAxis, Range1d, BoxAnnotation, Legend


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
airflow_sensors = client.data_sparql(airflow_query, sites=avail_sites)

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
        return None

    if sensor_metadata.shape[0] > 1:
        import pdb; pdb.set_trace()

    site = sensor_metadata.iloc[0]["site"]
    zone = sensor_metadata.iloc[0]["zone"]
    vav = sensor_metadata.iloc[0]["vav"]

    print(site, "|", vav, "|", zone, "\n")

    # get maximum and minimum cfm
    zone_num = zone.split("#")[1].lower().replace("rm", "")

    room_exists = vav_details[site]["vav2room"]["room"].str.contains(zone_num)
    room_found = vav_details[site]["vav2room"].loc[room_exists, :]

    if room_found.empty:
        return None

    vav_details_exists = vav_details[site]["vav_design"]["vav"].str.contains(room_found.iloc[0]["vav"])
    vav_details_found = vav_details[site]["vav_design"].loc[vav_details_exists, :]

    if vav_details_found.empty:
        return None

    vav_max_cfm = vav_details_found.iloc[0]["maximum cfm"]
    vav_min_cfm = vav_details_found.iloc[0]["minimum cfm"]

    return vav_max_cfm, vav_min_cfm


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

    plot_name = '{}-timeseries.html'.format(vav_airflow.iloc[0]["id"])
    output_file(join('./', plot_name))
    save(p)

    return p


# unique_sensors = airflow_view["airflow"].unique()
unique_sensors = airflow_sensors.data["id"].unique()

for sensor in unique_sensors:
    vav_max_cfm, vav_min_cfm = find_min_max_design_airflow(sensor, airflow_view, vav_details)
    vav_airflow = airflow_sensors.data.loc[airflow_sensors.data["id"].isin([sensor]), :]

    # plot = plot_airflow_dat(vav_airflow, vav_max_cfm, vav_min_cfm)



