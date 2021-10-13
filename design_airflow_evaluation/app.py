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

avail_sites = ['GHA_ICS', 'HART']
airflow_view = client.sparql(airflow_query)

select_sites = airflow_view["airflow"].str.contains("|".join(avail_sites).upper())
af_select_sites = airflow_view.loc[select_sites, :].sort_values(by=["airflow"]).reset_index(drop=True)


row = af_select_sites.loc[170]

airflow_dat = client.data_uris(row["airflow"])

################
################
### Clean Data
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