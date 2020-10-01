# @author: Carlos Duarte <cduarte@berkeley.edu>

import pymortar
import os
import pandas as pd

def query_and_qualify():
    """
    Build query to return zone air temperature measurements and qualify
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

    # save query
    query['query'] = sensor_query
    query['sensor'] = sensor

    print("Running on {0} sites".format(len(qualify_resp.sites)))

    return qualify_resp, query