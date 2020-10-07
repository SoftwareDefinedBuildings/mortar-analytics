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