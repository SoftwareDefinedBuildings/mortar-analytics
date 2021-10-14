import pymortar
import brickschema
import io

MORTAR_URL = "https://beta-api.mortardata.org"

# connect to Mortar client
client = pymortar.Client(MORTAR_URL)

# initialize container for query information
query = dict()

airflow_query = """SELECT ?airflow ?vav WHERE { 
    ?airflow    rdf:type                    brick:Supply_Air_Flow_Sensor .
    ?vav        rdf:type                    brick:VAV .
    ?vav        brick:hasPoint              ?airflow .
}"""

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
airflow_sensors = client.data_sparql(airflow_query, source=avail_sites)

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
# get building brick graphs
hart_model_file = './brick_models/hart_model.ttl'
gha_model_file = './brick_models/gha_model.ttl'

g_hart = client.get_graph('hart')
g_gha = client.get_graph('gha_ics')

g = brickschema.Graph(brick_version="1.2")
g_gha_brick = g.parse(g_gha, format="ttl")

g = brickschema.Graph(brick_version="1.2")
g_hart_brick = g.parse(io.BytesIO(g_hart), format="turtle")

# save graphs
with open(hart_model_file, 'wb') as f:
    f.write(g_hart)

with open(gha_model_file, 'wb') as f:
    f.write(g_gha)


g = brickschema.Graph(brick_version="1.2")
g_hart_model = g.load_file(hart_model_file)

g = brickschema.Graph(brick_version="1.2")
g_gha_model = g.load_file(gha_model_file, format="ttl")


################
################
### Evaluate Zone Airflow
################
################

unique_sensors = airflow_sensors.data["id"].unique()

for sensor in unique_sensors:
    sensor_metadata = airflow_view.loc[airflow_view["airflow"].str.contains(sensor), :]