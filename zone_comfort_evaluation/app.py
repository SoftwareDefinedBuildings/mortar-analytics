# @author: Carlos Duarte <cduarte@berkeley.edu>

import pymortar
import os
import pandas as pd


def query_and_qualify():
    """
    Build query to return zone air temperature measurements and qualify
    which site can run this application

    Parameters
    ----------
    None

    Returns
    -------
    qualify_resp : Mortar QualifyResponse object

    query : dictionary with query and sensor
    """
    # connect to client
    client = pymortar.Client()

    # container for query information
    query = dict()

    # define sensor type and query
    sensor = "Zone_Air_Temperature"

    sensor_query = """SELECT ?sensor WHERE {{
        ?sensor    rdf:type/rdfs:subClassOf*     brick:{0}_Sensor .
    }};""".format(sensor)

    # find sites that have the zone air temperature
    qualify_resp = client.qualify([sensor_query])
    if qualify_resp.error != "":
        print("ERROR: ", qualify_resp.error)
        os.exit(1)

    # save query and sensor used
    query['query'] = sensor_query
    query['sensor'] = sensor

    print("Running on {0} sites".format(len(qualify_resp.sites)))

    return qualify_resp, query

def fetch(qualify_resp, query, eval_start_time, eval_end_time):
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

    Returns
    -------
    fetch_resp : Mortar FetchResponse object

    """
    # get sensor and query to build fetch
    sensor          = query['sensor']
    sensor_query    = query['query']

    request = pymortar.FetchRequest(
        sites = qualify_resp.sites,
        views = [
            pymortar.View(
                name = "{}_sensors".format(sensor),
                definition = sensor_query,
            )
        ],
        dataFrames = [
            pymortar.DataFrame(
                name = "sensors",
                aggregation = pymortar.MEAN,
                window = "30m",
                timeseries = [
                    pymortar.Timeseries(
                        view = "{}_sensors".format(sensor),
                        dataVars = ["?sensor"],
                    )
                ]
            )
        ],
        time = pymortar.TimeParams(
            start = eval_start_time,
            end = eval_end_time
        )
    )

    client      = pymortar.Client()
    fetch_resp  = client.fetch(request)
    print(fetch_resp)

    import pdb; pdb.set_trace()

    return fetch_resp

if __name__ == "__main__":
    # define parameters
    eval_end_time = "2018-07-01T00:00:00Z"
    eval_start_time  = "2018-11-01T00:00:00Z"

    # Run the app
    qualify_resp, query = query_and_qualify()
    import pdb; pdb.set_trace()
    fetch_resp = fetch(qualify_resp, query, eval_start_time, eval_end_time)