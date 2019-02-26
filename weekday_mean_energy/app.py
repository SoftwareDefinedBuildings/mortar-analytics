import pymortar
import os
import pandas as pd

# tracks the mean energy use per daty of the week per site
mean_site_data = {}

def meter_for_site(resp, site):
    """
    Returns a dataframe view containing the building meters for the given site as columns, identified by UUID
    """
    uuids = [r[0] for r in resp.query("select meter_uuid from meters where site = '{0}'".format(site))]
    return pd.DataFrame(resp['meters'][uuids])

def normalize_meter_df(df):
    """
    Normalize the meter dataframe!
    - sum all meters together
    - convert to local timezone
    - name the 1 remaining column 'meter' so we don't have to remember UUIDs
    """
    df = pd.DataFrame(df.sum(axis=1))
    df.columns=['meter']
    df = df.set_index(df.index.tz_localize('UTC').tz_convert('US/Pacific'))
    return df

def drop_zeros(df):
    """
    Return a copy of the dataframe that doesn't contain zeros
    """
    return df.where(df > 0).dropna()

def pct_n(pct):
    """
    Returns a function that computes the Nth percentile of a series
    """
    def f(data):
        return pd.np.percentile(data, pct)
    return f

def process_site(resp, site):
    print("Processing", site)
    if site == 'hayward-station-8': # data is broken. #TODO: remove!
        return
    df = meter_for_site(resp, site)
    df = normalize_meter_df(df)
    df = drop_zeros(df)
    # apply weekday label
    df.loc[:, 'weekday'] = df.index.map(lambda date: date.day_name())
    
    # get meter data grouped by day of the week
    groups = df.groupby('weekday')
    
    by_week = {}

    for group in groups:
        day_name = group[0]
        gdf = group[1]
        gdf.loc[:, 'timeofday'] = gdf.index.strftime("%H:%M:%S")
#         gp = gdf.pivot_table(values='meter',  index=['timeofday'], aggfunc=[pd.np.mean, pd.np.max, pd.np.min, pct_n(95)])
        gp = gdf.pivot_table(values='meter',  index=['timeofday'], aggfunc=pd.np.mean)

        by_week[day_name] = gp
    
    mean_site_data[site] = by_week


# use default values (environment variables):
# MORTAR_API_ADDRESS: mortardata.org:9001
# MORTAR_API_USERNAME: required
# MORTAR_API_PASSWORD: required
client = pymortar.Client({})
meter_query = "SELECT ?meter WHERE { ?meter rdf:type/rdfs:subClassOf* brick:Building_Electric_Meter };"


#### QUALIFY Stage

qualify_resp = client.qualify([meter_query])
if qualify_resp.error != "":
    print("ERROR: ", qualify_resp.error)
    os.exit(1)

print("running on {0} sites".format(len(qualify_resp.sites)))

#### FETCH Stage
request = pymortar.FetchRequest(
    sites=qualify_resp.sites,
    views=[
        # defining relational table for the contents of the query (+site +meter_uuid columns)
        pymortar.View(
            name="meters",
            definition=meter_query,
        )
    ],
    dataFrames=[
        # 15min mean meter data
        pymortar.DataFrame(
            name="meters",
            aggregation=pymortar.MEAN,
            window="15m",
            timeseries=[
                pymortar.Timeseries(
                    view="meters",
                    dataVars=["?meter"]
                )
            ]
        )
    ],
    time=pymortar.TimeParams(
        start="2016-01-01T00:00:00Z",
        end="2019-01-01T00:00:00Z",
    )
)

resp = client.fetch(request)

# compute daily mean energy usage per day of the week for each site
for site in qualify_resp.sites:
    process_site(resp, site)

# convert the dataframe into 
means = {}
for site, weekdaydata in mean_site_data.items():
    means[site] = {}
    for weekday, df in weekdaydata.items():
        means[site][weekday] = df['meter'].mean()

# displays the 
m = pd.DataFrame(means)
m.style.background_gradient(cmap='RdYlGn_r',axis=0)
