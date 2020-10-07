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