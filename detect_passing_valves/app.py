import pymortar
import sys

# define parameters
eval_start_time  = "2018-06-01T00:00:00Z"
eval_end_time    = "2018-06-30T00:00:00Z"


client = pymortar.Client()

# define query to return valves

# returns equipments with valves including ahus
equip_query = """SELECT ?vlv ?ahu ?vlv_subclass ?equip ?equip_subclass ?sensor ?sensor_subclass 
WHERE {
    ?vlv    rdf:type/rdfs:subClassOf*   brick:Valve_Command .
    ?vlv    bf:isPointOf                ?equip .
    ?vlv    rdf:type                    ?vlv_subclass .
    ?equip  bf:hasPoint                 ?sensor .
    ?sensor rdf:type/rdfs:subClassOf*   brick:Temperature_Sensor .
    ?sensor rdf:type                    ?sensor_subclass .
    ?equip  rdf:type                    ?equip_subclass .
};"""

# returns supply air temps from ahu and vav and vav valve
vav_query = """SELECT *
WHERE {
    ?vav        rdf:type/rdfs:subClassOf? brick:VAV .
    ?vav        bf:isFedBy+                 ?ahu .
    ?ahu        bf:hasPoint                 ?ahu_supply .
    ?vav        bf:hasPoint                 ?vav_supply .
    ?ahu_supply rdf:type/rdfs:subClassOf*   brick:Supply_Air_Temperature_Sensor .
    ?vav_supply rdf:type/rdfs:subClassOf*   brick:Supply_Air_Temperature_Sensor .
    ?vav        bf:hasPoint                 ?vav_vlv .
    ?vav_vlv    rdf:type/rdfs:subClassOf*   brick:Valve_Command .
};"""

# find sites with these sensors and setpoints
qualify_resp = client.qualify([equip_query, vav_query])
if qualify_resp.error != "":
    print("ERROR: ", qualify_resp.error)
    sys.exit(1)
elif len(qualify_resp.sites) == 0:
    print("NO SITES RETURNED")
    sys.exit(0)

print("running on {0} sites".format(len(qualify_resp.sites)))

# build the fetch request
# request = pymortar.FetchRequest(
#     sites=qualify_resp.sites,
#     views=[
#         pymortar.View(
#             name="valves",
#             definition=query,
#         ),
#     ],
#     dataFrames=[
#         pymortar.DataFrame(
#             name="valves",
#             aggregation=pymortar.MEAN,
#             window="15m",
#             timeseries=[
#                 pymortar.Timeseries(
#                     view="valves",
#                     dataVars=["?vlv"],
#                 )
#             ]
#         ),
#     ],
#     time=pymortar.TimeParams(
#         start=eval_start_time,
#         end=eval_end_time,
#     )
# )

request = pymortar.FetchRequest(
    sites=qualify_resp.sites,
    views=[
        pymortar.View(
            name="all_equip",
            definition=equip_query,
        ),
        pymortar.View(
            name="vav_equip",
            definition=vav_query,
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
print(fetch_resp.view('vav_equip'))


# print the different types of valves in the data
#print(fetch_resp.view('valves').groupby(['vlv_subclass']).count())


import pdb; pdb.set_trace()