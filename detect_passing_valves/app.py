import pymortar
import sys

# define parameters
eval_start_time  = "2018-06-01T00:00:00Z"
eval_end_time    = "2018-06-30T00:00:00Z"


client = pymortar.Client()

# define query to return valves
# returns supply air temps from ahu and vav and vav valve
vav_query = """SELECT *
WHERE {
    ?vav        rdf:type/rdfs:subClassOf? brick:VAV .
    ?vav        bf:isFedBy+                 ?ahu .
    ?vav_vlv    rdf:type                    ?vlv_type .
    ?ahu        bf:hasPoint                 ?ahu_supply .
    ?vav        bf:hasPoint                 ?vav_supply .
    ?ahu_supply rdf:type/rdfs:subClassOf*   brick:Supply_Air_Temperature_Sensor .
    ?vav_supply rdf:type/rdfs:subClassOf*   brick:Supply_Air_Temperature_Sensor .
    ?vav        bf:hasPoint                 ?vav_vlv .
    ?vav_vlv    rdf:type/rdfs:subClassOf*   brick:Valve_Command .
};"""

# find sites with these sensors and setpoints
qualify_resp = client.qualify([vav_query])
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
            definition=vav_query,
        ),
    ],
    dataFrames=[
        pymortar.DataFrame(
            name="valve",
            aggregation=pymortar.MEAN,
            window="15m",
            timeseries=[
                pymortar.Timeseries(
                    view="valves",
                    dataVars=["?vav_vlv"],
                )
            ]
        ),
        pymortar.DataFrame(
            name="vav_temp",
            aggregation=pymortar.MEAN,
            window="15m",
            timeseries=[
                pymortar.Timeseries(
                    view="valves",
                    dataVars=["?vav_supply"],
                )
            ]
        ),
        pymortar.DataFrame(
            name="ahu_temp",
            aggregation=pymortar.MEAN,
            window="15m",
            timeseries=[
                pymortar.Timeseries(
                    view="valves",
                    dataVars=["?ahu_supply"],
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
print(fetch_resp.view('valves'))


# print the different types of valves in the data
#print(fetch_resp.view('valves').groupby(['vlv_subclass']).count())


import pdb; pdb.set_trace()