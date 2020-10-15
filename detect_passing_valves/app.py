import pymortar
import sys

# define parameters
eval_start_time  = "2018-06-01T00:00:00Z"
eval_end_time    = "2018-06-30T00:00:00Z"


client = pymortar.Client()

# define query to return valves
query = """SELECT ?vlv ?equip ?subclass WHERE {
    ?vlv    rdf:type/rdfs:subClassOf?   brick:Valve_Command .
    ?vlv    bf:isPointOf    ?equip .
    ?vlv    rdf:type ?subclass .
};"""

# find sites with these sensors and setpoints
qualify_resp = client.qualify([query])
if qualify_resp.error != "":
    print("ERROR: ", qualify_resp.error)
    sys.exit(1)
elif len(qualify_resp.sites) == 0:
    print("NO SITES RETURNED")
    sys.exit(0)

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
            name="valves",
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