import pymortar
import os

# define parameters
eval_start_time  = "2018-06-01T00:00:00Z"
eval_end_time    = "2018-06-30T00:00:00Z"


client = pymortar.Client()

# define query to return valves
query = """SELECT ?vlv ? equip WHERE {
    ?vlv rdf:type/rdfs:subClassOf*   brick:Valve_Command .
    OPTIONAL {
        ?vlv    bf:isPointOf    ?equip
    }
};"""

# find sites with these sensors and setpoints
qualify_resp = client.qualify([query])
if qualify_resp.error != "":
    print("ERROR: ", qualify_resp.error)
    os.exit(1)

print("running on {0} sites".format(len(qualify_resp.sites)))

# build the fetch request
request = pymortar.FetchRequest(
    sites=qualify_resp.sites,
    views=[
        pymortar.View(
            name="valves",
            definition=query,
        ),
    ],
    dataFrames=[
        pymortar.DataFrame(
            name="Vlv",
            aggregation=pymortar.MEAN,
            window="15m",
            timeseries=[
                pymortar.Timeseries(
                    view="valves",
                    dataVars=["?vlv"],
                )
            ]
        ),
    ],
    time=pymortar.TimeParams(
        start=eval_start_time,
        end=eval_end_time,
    )
)

# call the fetch api
fetch_resp = client.fetch(request)
print(fetch_resp)

import pdb; pdb.set_trace()