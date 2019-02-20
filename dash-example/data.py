import pymortar
import os
import pandas as pd

# use default values (environment variables):
# MORTAR_API_ADDRESS: mortardata.org:9001
# MORTAR_API_USERNAME: required
# MORTAR_API_PASSWORD: required
client = pymortar.Client({})

meter_query = "SELECT ?meter WHERE { ?meter rdf:type/rdfs:subClassOf* brick:Building_Electric_Meter };"

# run qualify stage to get list of sites with electric meters
resp = client.qualify([meter_query])
if resp.error != "":
    print("ERROR: ", resp.error)
    os.exit(1)

print("running on {0} sites".format(len(resp.sites)))

# define the view of meters (metadata)
meters = pymortar.View(
    sites=resp.sites,
    name="meters",
    definition=meter_query,
)

# define the meter timeseries streams we want
meter_data = pymortar.DataFrame(
    name="meters",
    aggregation=pymortar.MEAN,
    window="1h",
    timeseries=[
        pymortar.Timeseries(
            view="meters",
            dataVars=["?meter"]
        )
    ]
)

# temporal parameters for the query: 2017-2018 @ 15min mean
time_params = pymortar.TimeParams(
    start="2015-01-01T00:00:00Z",
    end="2018-01-01T00:00:00Z",
)

# form the full request object
request = pymortar.FetchRequest(
    sites=resp.sites,
    views=[meters],
    dataFrames=[meter_data],
    time=time_params
)

# download the data
print("Starting to download data...")
data = client.fetch(request)

# compute daily min/max/mean for each site
ranges = []
for site in resp.sites:
    meter_uuids = data.query("select meter_uuid from meters where site='{0}'".format(site))
    meter_uuids = [row[0] for row in meter_uuids]
    meterdf = data['meters'][meter_uuids].sum(axis=1)
    ranges.append( [site, meterdf.min(), meterdf.max(), meterdf.mean()])

site_summary = pd.DataFrame.from_records(ranges)
site_summary.columns = ['site','min_daily','max_daily','mean_daily']
