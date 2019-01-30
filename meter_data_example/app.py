import pymortar
import os

# use default values (environment variables):
# MORTAR_API_ADDRESS: mortardata.org:9001
# MORTAR_API_USERNAME: required
# MORTAR_API_PASSWORD: required
client = pymortar.Client({})

meter_query = "SELECT ?meter WHERE { ?meter rdf:type/rdfs:subClassOf* brick:Electric_Meter };"

# run qualify stage to get list of sites with electric meters
resp = client.qualify([meter_query])
if resp.error != "":
    print("ERROR: ", resp.error)
    os.exit(1)

print("running on {0} sites".format(len(resp.sites)))

# define the meter stream
meter_stream = pymortar.Stream(
    name="meter",
    definition=meter_query,
    dataVars=["?meter"],
    aggregation=pymortar.MEAN,
)

# temporal parameters for the query: 2017-2018 @ 15min mean
time_params = pymortar.TimeParams(
    start="2016-01-01T00:00:00Z",
    end="2018-01-01T00:00:00Z",
    window="1h",
)

# form the full request object
request = pymortar.FetchRequest(
    sites=resp.sites,
    streams=[meter_stream],
    time=time_params
)

# download the data
print("Starting to download data...")
data = client.fetch(request)

print(data.df.describe())

# clean the dataframe by forward-filling in null values
data.df.fillna(method='ffill', inplace=True)

# process the meter data for each site individually.
# use the sql db to find the uuids for each site
for site in resp.sites:
    uuids = data.query("SELECT meter_uuid FROM meter where site='{0}';".format(site))
    uuids = [x[0] for x in uuids] # unpack from nested list

    # use uuids as column selectors
    # add together meters
    meter_sum = data.df[uuids].sum(axis=1)
    print("#### Describe Meter for site {0} ####".format(site))
    print(meter_sum.describe())
