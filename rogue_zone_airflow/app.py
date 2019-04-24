import pymortar
import os
import pandas as pd

client = pymortar.Client()

# define queries for airflow sensors and setpoints
air_flow_sensor_query = """SELECT ?sensor ?equip WHERE {
    ?sensor    rdf:type/rdfs:subClassOf*     brick:Air_Flow_Sensor .
    ?sensor    bf:isPointOf ?equip .
};"""
air_flow_setpoint_query = """SELECT ?sp ?equip WHERE {
    ?sp    rdf:type/rdfs:subClassOf*     brick:Air_Flow_Setpoint .
    ?sp    bf:isPointOf ?equip .
};"""

# find sites with these sensors and setpoints
qualify_resp = client.qualify([air_flow_sensor_query, air_flow_setpoint_query])
if qualify_resp.error != "":
    print("ERROR: ", qualify_resp.error)
    os.exit(1)
    
print("running on {0} sites".format(len(qualify_resp.sites)))

# define dataset. We are keeping the airflow sensors and setpoints separate for now
# because we will join using the Views later
request = pymortar.FetchRequest(
    sites=qualify_resp.sites,
    views=[
        pymortar.View(
            name="airflow_sensors",
            definition=air_flow_sensor_query,
        ),
        pymortar.View(
            name="airflow_sps",
            definition=air_flow_setpoint_query,
        )
    ],
    dataFrames=[
        pymortar.DataFrame(
            name="sensors",
            aggregation=pymortar.MEAN,
            window="30m",
            timeseries=[
                pymortar.Timeseries(
                    view="airflow_sensors",
                    dataVars=["?sensor"],
                )
            ]
        ),
        pymortar.DataFrame(
            name="setpoints",
            aggregation=pymortar.MEAN,
            window="30m",
            timeseries=[
                pymortar.Timeseries(
                    view="airflow_sps",
                    dataVars=["?sp"],
                )
            ]
        )
    ],
    time=pymortar.TimeParams(
        start="2018-01-01T00:00:00Z",
        end="2019-01-01T00:00:00Z",
    )
)

resp = client.fetch(request)
print(resp)

# get all the equipment we will run the analysis for. Equipment relates sensors and setpoints
equipment = [r[0] for r in resp.query("select distinct equip from airflow_sensors")]

# find airflow sensors that aren't just all zeros
valid_sensor_cols = (resp['sensors'] > 0).any().where(lambda x: x).dropna().index
sensor_df = resp['sensors'][valid_sensor_cols]
setpoint_df = resp['setpoints']

records = []

for idx, equip in enumerate(equipment):
    # for each equipment, pull the UUID for the sensor and setpoint
    q = """
    SELECT sensor_uuid, sp_uuid, airflow_sps.equip, airflow_sps.site
    FROM airflow_sensors
    LEFT JOIN airflow_sps
    ON airflow_sps.equip = airflow_sensors.equip
    WHERE airflow_sensors.equip = "{0}";
    """.format(equip)
    res = resp.query(q)
    if len(res) == 0:
        continue

    sensor_col = res[0][0]
    setpoint_col = res[0][1]
    
    if sensor_col is None or setpoint_col is None:
        continue

    if sensor_col not in sensor_df:
        print('no sensor', sensor_col)
        continue
    
    if setpoint_col not in setpoint_df:
        print('no sp', setpoint_col)
        continue

    # create the dataframe for this pair of sensor and setpoint
    df = pd.DataFrame([sensor_df[sensor_col], setpoint_df[setpoint_col]]).T
    df.columns = ['airflow','setpoint']
    bad = (df.airflow + 10) < df.setpoint # by 10 cfm
    if len(df[bad]) == 0: continue
    df['same'] = bad.astype(int).diff(1).cumsum()
    # this increments every time we get a new run of sensor being below the setpoint
    # use this to group up those ranges
    df['same2'] = bad.astype(int).diff().ne(0).cumsum()

    lal = df[bad].groupby('same2')['same']
    # grouped by ranges that meet the predicate (df.airflow + 10 < df.setpoint)
    for g in lal.groups:
        idx = list(lal.groups[g])
        if len(idx) < 2: continue
        data = df[idx[0]:idx[-1]]
        if len(data) >= 4: # 2 hours
            fmt = {
                'site': res[0][3],
                'equip': equip,
                'hours': len(data) / 2,
                'start': idx[0],
                'end': idx[1],
                'diff': (data['setpoint'] - data['airflow']).mean(),
            }
            records.append(fmt)
            print("Low Airflow for {hours} hours From {start} to {end}, avg diff {diff:.2f}".format(**fmt))
r = pd.DataFrame(records)
print('## RESULTS ##')
print(r)
r.to_csv('rogue_zones.csv', index=False)
