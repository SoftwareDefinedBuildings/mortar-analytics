__author__ = "Anand Krishnan Prakash"
__email__ = "akprakash@lbl.gov"

import pymortar
import datetime
import pandas as pd


RESAMPLE_MINUTES = 60
FILENAME = "simultaneous_heat_cool_zones.csv"
START_TIME = "2018-12-10T00:00:00Z"
END_TIME = "2019-01-01T00:00:00Z"


def get_error_message(x):
    dt_format = "%Y-%m-%d %H:%M:%S"
    st = x.name
    st_str = st.strftime(dt_format)
    et_str = (st+datetime.timedelta(minutes=RESAMPLE_MINUTES)).strftime(dt_format)
    site = x.site
    room = x.room
    zone = x.zone
    heat_percent = round(x.heat_percent, 2)
    cool_percent = round(x.cool_percent, 2)
    msg = "From {0} to {1}, zone: \'{2}\' in room: \'{3}\' at site: \'{4}\', was heating for {5}% of the time and cooling for {6}% of the time".format(
            st_str,
            et_str,
            zone,
            room,
            site,
            heat_percent,
            cool_percent
    )
    return msg

client = pymortar.Client({})

tstat_query = """
    SELECT ?tstat ?room ?zone ?state ?temp ?hsp ?csp WHERE {
        ?tstat bf:hasLocation ?room .
        ?zone bf:hasPart ?room .

        ?tstat bf:hasPoint ?state .
        ?tstat bf:hasPoint ?temp .
        ?tstat bf:hasPoint ?hsp .
        ?tstat bf:hasPoint ?csp .

        ?zone rdf:type/rdfs:subClassOf* brick:Zone .
        ?tstat rdf:type/rdfs:subClassOf* brick:Thermostat .
        ?state rdf:type/rdfs:subClassOf* brick:Thermostat_Status .
        ?temp  rdf:type/rdfs:subClassOf* brick:Temperature_Sensor  .
        ?hsp   rdf:type/rdfs:subClassOf* brick:Supply_Air_Temperature_Heating_Setpoint .
        ?csp   rdf:type/rdfs:subClassOf* brick:Supply_Air_Temperature_Cooling_Setpoint .
    };
"""
qualify_response = client.qualify([tstat_query])
if qualify_response.error != "":
    print("ERROR: ", qualify_response.error)
    os.exit(1)

print("Running on {0} sites".format(len(qualify_response.sites)))


tstat_view = pymortar.View(
    name="tstat_points",
    sites=qualify_response.sites,
    definition=tstat_query,
)

tstat_streams = pymortar.DataFrame(
    name="thermostat_data",
    aggregation=pymortar.MAX,
    window="1m",
    timeseries=[
        pymortar.Timeseries(
            view="tstat_points",
            dataVars=["?state", "?temp", "?hsp", "?csp"]
        )
    ]
)

time_params = pymortar.TimeParams(
    start=START_TIME,
    end=END_TIME
)

request = pymortar.FetchRequest(
    sites=qualify_response.sites, # from our call to Qualify
    views=[
        tstat_view
    ],
    dataFrames=[
        tstat_streams
    ],
    time=time_params
)
result = client.fetch(request)

tstat_df = result['thermostat_data']
tstats = [tstat[0] for tstat in result.query("select tstat from tstat_points")]


error_df_list = []
for tstat in tstats:
    q = """
            SELECT state_uuid, temp_uuid, hsp_uuid, csp_uuid, room, zone, site
            FROM tstat_points
            WHERE tstat = "{0}";
        """.format(tstat)
    res = result.query(q)
    
    if len(res) == 0:
        continue

    state_col, iat_col, hsp_col, csp_col, room, zone, site = res[0]
    df = tstat_df[[state_col, iat_col, hsp_col, csp_col]]
    df.columns = ['state',  'iat', 'hsp', 'csp']
    
    df2 = pd.DataFrame()
    resample_time = '{0}T'.format(RESAMPLE_MINUTES)
    df2['min_hsp'] = df['hsp'].resample(resample_time).min()
    df2['min_csp'] = df['csp'].resample(resample_time).min()
    df2['max_hsp'] = df['hsp'].resample(resample_time).max()
    df2['max_csp'] = df['csp'].resample(resample_time).max()    

    df2['heat_percent'] = df['state'].resample(resample_time).apply(lambda x: ((x==1).sum() + (x==4).sum())/RESAMPLE_MINUTES*100)
    df2['cool_percent'] = df['state'].resample(resample_time).apply(lambda x: ((x==2).sum() + (x==5).sum())/RESAMPLE_MINUTES*100)
    
    df2['tstat'] = tstat
    df2['room'] = room.split('#')[1]
    df2['zone'] = zone.split('#')[1]
    df2['site'] = site
        
    df2['both_heat_cool'] = False
    df2.loc[((df2.heat_percent > 0) & (df2.cool_percent > 0)), 'both_heat_cool'] = True
    if not df2[df2['both_heat_cool'] == True].empty:
        error_df_list.append(df2[df2['both_heat_cool'] == True])

error_df = pd.concat(error_df_list, axis=0)[['site', 'zone', 'room', 'heat_percent', 'cool_percent', 'min_hsp', 'min_csp', 'max_hsp', 'max_csp']]
error_df.index.name = 'time'
error_msgs = error_df.apply(lambda x: get_error_message(x), axis=1).values
for msg in error_msgs:
	print(msg)

print("Writing results to {0}".format(FILENAME))
error_df.to_csv(FILENAME)
