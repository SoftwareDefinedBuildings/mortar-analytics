import pymortar
import os
import pandas as pd
import numpy as np

MORTAR_URL = ""

def _query_and_qualify(sensor):
    """
    Build query to return zone air temperature measurements and qualify
    which site can run this application

    Parameters
    ----------
    sensor : sensor name type to evaluate e.g. Zone_Air_Temperature

    Returns
    -------
    qualify_resp : Mortar QualifyResponse object

    query : dictionary with query and sensor
    """
    # connect to client
    client = pymortar.Client(MORTAR_URL)

    # initialize container for query information
    query = dict()

    # define queries for input sensors and setpoints
    sensor_query = """SELECT ?sensor ?equip WHERE {{
        ?sensor    rdf:type/rdfs:subClassOf*     brick:{0}_Sensor .
        ?sensor    brick:isPointOf ?equip .
    }}""".format(sensor)

    setpoint_query = """SELECT ?sp ?equip WHERE {{
        ?sp    rdf:type/rdfs:subClassOf*     brick:{0}_Setpoint .
        ?sp    brick:isPointOf ?equip .
    }}""".format(sensor)

    # find sites with input sensors and setpoints
    qualify_resp = client.qualify({"measurement": sensor_query, "setpoint": setpoint_query})
    # if qualify_resp.error != "":
    #     print("ERROR: ", qualify_resp.error)
    #     os.exit(1)

    # save queries and sensor information
    query['query'] = dict()
    query['query']['sensor'] = sensor_query
    query['query']['setpoint'] = setpoint_query
    query['sensor'] = sensor

    print("running on {0} sites".format(len(qualify_resp.sites)))
    print(qualify_resp.sites)

    return qualify_resp, query


def _fetch(qualify_resp, query, eval_start_time, eval_end_time, window=15):
    """
    Build the fetch query and define the thermal comfort evaluation time.

    Parameters
    ----------
    qualify_resp : Mortar QualifyResponse object

    query : dictionary with query and sensor

    eval_start_time : start date and time in format (yyyy-mm-ddTHH:MM:SSZ) for the thermal
                      comfort evaluation period

    eval_end_time : end date and time in format (yyyy-mm-ddTHH:MM:SSZ) for the thermal
                    comfort evaluation period

    window : aggregation window in minutes to average the measurement data

    Returns
    -------
    fetch_resp : Mortar FetchResponse object

    """
    # connect to client
    client = pymortar.Client(MORTAR_URL)

    sensor         = query['sensor']
    sensor_query   = query['query']['sensor']
    setpoint_query = query['query']['setpoint']

    # build the fetch request
    avail_sites = qualify_resp.sites

    # request = pymortar.FetchRequest(
    #     sites=req_sites,
    #     views=[
    #         pymortar.View(
    #             name="{}_sensors".format(sensor),
    #             definition=sensor_query,
    #         ),
    #         pymortar.View(
    #             name="{}_sps".format(sensor),
    #             definition=setpoint_query,
    #         )
    #     ],
    #     dataFrames=[
    #         pymortar.DataFrame(
    #             name="sensors",
    #             aggregation=pymortar.MEAN,
    #             window="{}m".format(window),
    #             timeseries=[
    #                 pymortar.Timeseries(
    #                     view="{}_sensors".format(sensor),
    #                     dataVars=["?sensor"],
    #                 )
    #             ]
    #         ),
    #         pymortar.DataFrame(
    #             name="setpoints",
    #             aggregation=pymortar.MEAN,
    #             window="{}m".format(window),
    #             timeseries=[
    #                 pymortar.Timeseries(
    #                     view="{}_sps".format(sensor),
    #                     dataVars=["?sp"],
    #                 )
    #             ]
    #         )
    #     ],
    #     time=pymortar.TimeParams(
    #         start=eval_start_time,
    #         end=eval_end_time,
    #     )
    # )

    # call the fetch api
    # fetch_resp = client.fetch(request)
    # print(fetch_resp)

    # fetch point metadata
    sensor_view = client.sparql(sensor_query, sites=avail_sites).reset_index(drop=True)
    setpoint_view = client.sparql(setpoint_query, sites=avail_sites).reset_index(drop=True)

    fetch_resp = dict()
    fetch_resp['sensor'] = sensor_view
    fetch_resp['setpoint'] = setpoint_view

    # save time parameters
    fetch_resp['time'] = dict()
    fetch_resp['time']['start'] = eval_start_time
    fetch_resp['time']['end'] = eval_end_time
    fetch_resp['time']['window'] = window

    return fetch_resp

def _clean_metadata(fetch_resp):
    """
    Clean data by deleting streams with zero values.

    Parameters
    ----------
    sensor : sensor name type to evaluate e.g. Zone_Air_Temperature

    fetch_resp : Mortar FetchResponse object

    Returns
    -------
    sensor_df : dataframe of nonzero sensor measurements

    setpoint_df : dataframe of setpoint values

    equipment : equipment related to the sensor measurement

    """
    # get all the equipment we will run the analysis for. Equipment relates sensors and setpoints
    sensor_view = fetch_resp['sensor']
    setpoint_view = fetch_resp['setpoint']

    comb_metadata = pd.merge(sensor_view, setpoint_view, how='outer', on=['equip', 'site'])
    comb_metadata = comb_metadata.dropna().reset_index(drop=True)

    # equipment = [r[0] for r in fetch_resp.query("select distinct equip from {}_sensors".format(sensor))]

    # # find sensor measurements that aren't just all zeros
    # valid_sensor_cols   = (fetch_resp['sensors'] > 0).any().where(lambda x: x).dropna().index
    # sensor_df           = fetch_resp['sensors'][valid_sensor_cols]
    # setpoint_df         = fetch_resp['setpoints']

    return comb_metadata

def _clean_df(sensor_df, setpoint_df):
    """
    Match and clean data from sensor and setpoints
    """
    df = pd.merge(sensor_df, setpoint_df, how='outer', on=['time'])
    df = df.dropna().reset_index(drop=True)
    df = df.rename(columns={'value_x': 'sensor_val', 'id_x': 'sensor_id', 'value_y': 'setpoint_val', 'id_y': 'setpoint_id'})

    return df

def _analyze(query, fetch_resp, th_type='abs', th_diff=0.25, th_time=15):
    """
    Parameters
    ----------
    query : dictionary with query and sensor

    fetch_resp : Mortar FetchResponse object

    th_type : Type of comparison performed when evaluating sensor measurement against the setpoint value.
              Available options are (any input value within list is valid):
                ['under', 'u', '-', 'neg', '<'] = return sensors that are under setpoint by th_diff for th_time
                ['over', 'o', '+', 'pos', '>']  = return sensors that are over setpoint by th_diff for th_time
                ['outbound', 'outbounds', 'ob', '><'] = return sensors that are either under minimum setpoint value by th_diff
                                                    or over maximum setpoint value by th_diff for th_time
                ['bounded', 'inbounds','inbound', 'ib', '<>'] = return sensors that are within minimum setpoint value + th_diff
                                                                and maximum setpoint value - th_diff
                ['abs', ''] (default type) = return sensors that are +/- th_diff of setpoint value.

    th_diff: threshold allowance for determining if sensor measurement is not adhereing to setpoint
             in the same units of selected sensor e.g. if 'over' is selected for th_type and 2 for
             th_diff then 'bad sensors' will return whenever sensor measurement is setpoint + 2.

    th_time : Amount of time in minutes that a sensor measurment needs to meet the selected criteria in order to qualify as 'bad'.
             Must be greater or equal and a multiple of the data aggregation window.

    window : aggregation window in minutes that the data from sensors and setpoint are in

    Returns
    -------
    None

    The app produces a CSV file called `<sensor>_measure_vs_setpoint_<type of analysis>.csv` when run
        where '<sensor>' states the sensor type and '<analysis>' states the type of analysis performed.

    """
    # connect to client
    client = pymortar.Client(MORTAR_URL)

    sensor  = query['sensor']
    comb_metadata = _clean_metadata(fetch_resp)

    start = fetch_resp['time']['start']
    end = fetch_resp['time']['end']
    window = fetch_resp['time']['window']
    records = []

    for idxi, row in comb_metadata.iterrows():
        # get data for sensor and setpoint
        sensor_df = client.data_uris([row['sensor']], start=start, end=end, agg='mean', window="{}m".format(window))
        setpoint_df = client.data_uris([row['sp']], start=start, end=end, agg='mean', window="{}m".format(window))

        if not any([sensor_df.data.empty, setpoint_df.data.empty]):
            df = _clean_df(sensor_df.data, setpoint_df.data)

            if True:
                zone_name = str(row['sensor']).split('#')[1]
                csv_name = f"./zone_dat/{row['site']}-{zone_name}-{idxi}.csv"

                with open(csv_name, 'w') as fout:
                    fout.write(f"sensor: {row['sensor']}\n")
                    fout.write(f"setpoint {row['sp']}\n\n")

                    df.drop(columns=['sensor_id', 'setpoint_id']).to_csv(fout, index=False)
                fout.close
        else:
            continue

        # # for each equipment, pull the UUID for the sensor and setpoint
        # q = """
        # SELECT sensor_uuid, sp_uuid, {1}_sps.equip, {1}_sps.site
        # FROM {1}_sensors
        # LEFT JOIN {1}_sps
        # ON {1}_sps.equip = {1}_sensors.equip
        # WHERE {1}_sensors.equip = "{0}";
        # """.format(equip, sensor)

        # res = fetch_resp.query(q)
        # if len(res) == 0:
        #     continue

        # sensor_col = res[0][0]
        # setpoint_col = res[0][1]

        # if sensor_col is None or setpoint_col is None:
        #     continue

        # if sensor_col not in sensor_df:
        #     print('no sensor', sensor_col)
        #     continue

        # if setpoint_col not in setpoint_df:
        #     print('no sp', setpoint_col)
        #     continue

        # # create the dataframe for this pair of sensor and setpoint
        # df = pd.DataFrame([sensor_df[sensor_col], setpoint_df[setpoint_col]]).T
        # df.columns = ["{}_sensors".format(sensor), "{}_sps".format(sensor)]

        if th_type in ['under', 'u', '-', 'neg', '<']: # if measurement is under sp by th_diff
            bad = (df["sensor_val"]) < (df["setpoint_val"] - th_diff)
            str_th_type = 'Undershooting'

        elif th_type in ['over', 'o', '+', 'pos', '>']: # if measurement is over sp by th_diff
            bad = (df["sensor_val"]) > (df["setpoint_val"] + th_diff)
            str_th_type = 'Overshooting'

        elif th_type in ['outbound', 'outbounds', 'ob', '><']: # if measurement is either below min sp or above max sp by th_diff
            max_sp = df["setpoint_val"].max()
            min_sp = df["setpoint_val"].min()

            bad_max = (df["sensor_val"]) > (max_sp + th_diff)
            bad_min = (df["sensor_val"]) < (min_sp - th_diff)

            bad = pd.DataFrame([bad_min, bad_max]).all()
            str_th_type = 'Exceedance_of_min-max'

        elif th_type in ['bounded', 'inbounds','inbound', 'ib', '<>']: # if measurement is either within min and max sp by th_diff
            max_sp = df["setpoint_val"].max()
            min_sp = df["setpoint_val"].min()

            bad_max = (df["sensor_val"]) < (max_sp - th_diff)
            bad_min = (df["sensor_val"]) > (min_sp + th_diff)

            bad = pd.DataFrame([bad_min, bad_max]).all()
            str_th_type = 'Within_min-max'

        else:
            bad = abs(df["sensor_val"] - df["setpoint_val"]) > th_diff
            str_th_type = 'Not_within_setpoint'

        if len(df[bad]) == 0: continue
        df['same'] = bad.astype(int).diff(1).cumsum()
        # this increments every time we get a new run of sensor being below the setpoint
        # use this to group up those ranges
        df['same2'] = bad.astype(int).diff().ne(0).cumsum()

        lal = df[bad].groupby('same2')['same']
        # grouped by ranges that meet the predicate (df.sensor + th_diff < df.setpoint)
        for g in lal.groups:
            idx = list(lal.groups[g])
            if len(idx) < 2: continue
            data = df[idx[0]:idx[-1]]
            if len(data) >= (60/th_time): # multiply by window frame to get hours
                fmt = {
                    'site': row['site'],
                    'equipment': row['equip'],
                    'hours': len(data) / (60/window),
                    'start': idx[0],
                    'end': idx[-1],
                    'sensor_val': (data["setpoint_val"]).mean(),
                    'setpoint_val': (data["sensor_val"]).mean(),
                    'diff': (data["setpoint_val"] - data["sensor_val"]).mean(),
                }
                records.append(fmt)
                print("{str_th_type} {sensor} for {hours} hours From {start} to {end}, avg diff {diff:.2f}".format(**fmt,
                                                                                                                   sensor=sensor,
                                                                                                                   str_th_type=str_th_type))

    r = pd.DataFrame(records)
    print('##### Saving Results #####')
    r.to_csv('{}_measure_vs_setpoint_{}.csv'.format(sensor, str_th_type), index=False)


def evaluate_sensors(sensor, eval_start_time, eval_end_time, th_type, th_diff, th_time, window):
    """
    Compare sensor measurements against their respective setpoint values

    Parameters
    ----------
    sensor : sensor name type to evaluate e.g. Zone_Air_Temperature

    eval_start_time : start date and time in format (yyyy-mm-ddTHH:MM:SSZ) for the thermal
                      comfort evaluation period

    eval_end_time : end date and time in format (yyyy-mm-ddTHH:MM:SSZ) for the thermal
                    comfort evaluation period

    th_type : Type of comparison performed when evaluating sensor measurement against the setpoint value.
              Available options are (any input value within list is valid):
                ['under', 'u', '-', 'neg', '<'] = return sensors that are under setpoint by th_diff for th_time
                ['over', 'o', '+', 'pos', '>']  = return sensors that are over setpoint by th_diff for th_time
                ['outbound', 'outbounds', 'ob', '><'] = return sensors that are either under minimum setpoint value by th_diff
                                                    or over maximum setpoint value by th_diff for th_time
                ['bounded', 'inbounds','inbound', 'ib', '<>'] = return sensors that are within minimum setpoint value + th_diff
                                                                and maximum setpoint value - th_diff
                ['abs', ''] (default type) = return sensors that are +/- th_diff of setpoint value.

    th_diff: threshold allowance for determining if sensor measurement is not adhereing to setpoint
             in the same units of selected sensor e.g. if 'over' is selected for th_type and 2 for
             th_diff then 'bad sensors' will return whenever sensor measurement is setpoint + 2.

    th_time : Amount of time in minutes that a sensor measurment needs to meet the selected criteria in order to qualify as 'bad'.
             Must be greater or equal and a multiple of the data aggregation window.

    window : aggregation window in minutes that the data from sensors and setpoint are in

    Returns
    -------
    None

    The app produces a CSV file called `<sensor>_measure_vs_setpoint_<type of analysis>.csv` when run
        where '<sensor>' states the sensor type and '<analysis>' states the type of analysis performed.
    Returns
    -------
    """

    # build query and determine which sites have the point to do this analysis
    qualify_resp, query = _query_and_qualify(sensor)

    # find sites with these sensors and setpoints or else exit
    # if qualify_resp.error != "":
    #     print("ERROR: ", qualify_resp.error)
    #     os.exit(1)

    # build the request to fetch data for qualified sites
    fetch_resp = _fetch(qualify_resp, query, eval_start_time, eval_end_time, window)

    # analyze and print out measurements/sensors that are not meeting its setpoints
    _analyze(query, fetch_resp, th_type, th_diff, th_time)

    print('##### App has finish evaluating sensors #####')

if __name__ == '__main__':
    # define input values
    sensor      = "Zone_Air_Temperature"
    eval_start_time  = None
    eval_end_time    = None
    th_diff     = 2
    th_time     = 30
    th_type     = 'abs'
    window      = 15

    # Run the app
    evaluate_sensors(sensor, eval_start_time, eval_end_time, th_type, th_diff, th_time, window)
