import pymortar
import os
import pandas as pd

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
    client = pymortar.Client()

    # initialize container for query information
    query = dict()

    # define queries for input sensors and setpoints
    sensor_query = """SELECT ?sensor WHERE {{
        ?sensor    rdf:type/rdfs:subClassOf*     brick:{0}_Sensor .
    }};""".format(sensor)

    setpoint_query = """SELECT ?sp WHERE {{
        ?sp    rdf:type/rdfs:subClassOf*     brick:{0}_Setpoint .
    }};""".format(sensor)

    # find sites with input sensors and setpoints
    qualify_resp = client.qualify([sensor_query, setpoint_query])
    if qualify_resp.error != "":
        print("ERROR: ", qualify_resp.error)
        os.exit(1)

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
    sensor         = query['sensor']
    sensor_query   = query['query']['sensor']
    setpoint_query = query['query']['setpoint']

    # build the fetch request
    request = pymortar.FetchRequest(
        sites=qualify_resp.sites,
        views=[
            pymortar.View(
                name="{}_sensors".format(sensor),
                definition=sensor_query,
            ),
            pymortar.View(
                name="{}_sps".format(sensor),
                definition=setpoint_query,
            )
        ],
        dataFrames=[
            pymortar.DataFrame(
                name="sensors",
                aggregation=pymortar.MEAN,
                window="{}m".format(window),
                timeseries=[
                    pymortar.Timeseries(
                        view="{}_sensors".format(sensor),
                        dataVars=["?sensor"],
                    )
                ]
            ),
            pymortar.DataFrame(
                name="setpoints",
                aggregation=pymortar.MEAN,
                window="{}m".format(window),
                timeseries=[
                    pymortar.Timeseries(
                        view="{}_sps".format(sensor),
                        dataVars=["?sp"],
                    )
                ]
            )
        ],
        time=pymortar.TimeParams(
            start=eval_start_time,
            end=eval_end_time,
        )
    )

    # call the fetch api
    client = pymortar.Client()
    fetch_resp = client.fetch(request)
    print(fetch_resp)

    return fetch_resp

def _clean(sensor, fetch_resp):
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
    equipment = [r[0] for r in fetch_resp.query("select distinct equip from {}_sensors".format(sensor))]

    # find sensor measurements that aren't just all zeros
    valid_sensor_cols   = (fetch_resp['sensors'] > 0).any().where(lambda x: x).dropna().index
    sensor_df           = fetch_resp['sensors'][valid_sensor_cols]
    setpoint_df         = fetch_resp['setpoints']

    return sensor_df, setpoint_df, equipment


def _analyze(query, fetch_resp, th_type='abs', th_diff=0.25, th_time=15, window=15):
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

    sensor  = query['sensor']
    sensor_df, setpoint_df, equipment = _clean(sensor, fetch_resp)
    records = []

    for idx, equip in enumerate(equipment):
        # for each equipment, pull the UUID for the sensor and setpoint
        q = """
        SELECT sensor_uuid, sp_uuid, {1}_sps.equip, {1}_sps.site
        FROM {1}_sensors
        LEFT JOIN {1}_sps
        ON {1}_sps.equip = {1}_sensors.equip
        WHERE {1}_sensors.equip = "{0}";
        """.format(equip, sensor)

        res = fetch_resp.query(q)
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
        df.columns = ["{}_sensors".format(sensor), "{}_sps".format(sensor)]

        if th_type in ['under', 'u', '-', 'neg', '<']: # if measurement is under sp by th_diff
            bad = (df["{}_sensors".format(sensor)]) < (df["{}_sps".format(sensor)] - th_diff)
            str_th_type = 'Undershooting'

        elif th_type in ['over', 'o', '+', 'pos', '>']: # if measurement is over sp by th_diff
            bad = (df["{}_sensors".format(sensor)]) > (df["{}_sps".format(sensor)] + th_diff)
            str_th_type = 'Overshooting'

        elif th_type in ['outbound', 'outbounds', 'ob', '><']: # if measurement is either below min sp or above max sp by th_diff
            max_sp = df["{}_sps".format(sensor)].max()
            min_sp = df["{}_sps".format(sensor)].min()

            bad_max = (df["{}_sensors".format(sensor)]) > (max_sp + th_diff)
            bad_min = (df["{}_sensors".format(sensor)]) < (min_sp - th_diff)

            bad = pd.DataFrame([bad_min, bad_max]).all()
            str_th_type = 'Exceedance_of_min-max'

        elif th_type in ['bounded', 'inbounds','inbound', 'ib', '<>']: # if measurement is either within min and max sp by th_diff
            max_sp = df["{}_sps".format(sensor)].max()
            min_sp = df["{}_sps".format(sensor)].min()

            bad_max = (df["{}_sensors".format(sensor)]) < (max_sp - th_diff)
            bad_min = (df["{}_sensors".format(sensor)]) > (min_sp + th_diff)

            bad = pd.DataFrame([bad_min, bad_max]).all()
            str_th_type = 'Within_min-max'

        else:
            bad = abs(df["{}_sensors".format(sensor)] - df["{}_sps".format(sensor)]) > th_diff
            str_th_type = 'Within_setpoint'

        if len(df[bad]) == 0: continue
        df['same'] = bad.astype(int).diff(1).cumsum()
        # this increments every time we get a new run of sensor being below the setpoint
        # use this to group up those ranges
        df['same2'] = bad.astype(int).diff().ne(0).cumsum()

        lal = df[bad].groupby('same2')['same']
        # grouped by ranges that meet the predicate (df.sensor + th_diff < df.setpoint)
        for g in lal.groups:
            idx = list(lal.groups[g])
            if len(idx) < (60/th_time): continue ## VERIFY/DEBUG this line
            data = df[idx[0]:idx[-1]]
            if len(data) >= (60/th_time): # multiply by window frame to get hours ## VERIFY/DEBUG this line
                fmt = {
                    'site': res[0][3],
                    'equipment': equip,
                    'hours': len(data) / (60/window), ## VERIFY/DEBUG this line
                    'start': idx[0],
                    'end': idx[-1],
                    'sensor_val': (data["{}_sps".format(sensor)]).mean(),
                    'setpoint_val': (data["{}_sensors".format(sensor)]).mean(),
                    'diff': (data["{}_sps".format(sensor)] - data["{}_sensors".format(sensor)]).mean(),
                }
                records.append(fmt)
                print("{str_th_type} {sensor} for {hours} hours From {start} to {end}, avg diff {diff:.2f}".format(**fmt,
                                                                                                                   sensor=sensor,
                                                                                                                   str_th_type=str_th_type))
    r = pd.DataFrame(records)
    print('##### Saving Results #####')
    r.to_csv('{}_measure_vs_setpoint_{}.csv'.format(sensor, str_th_type), index=False)