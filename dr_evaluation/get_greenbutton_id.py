import pymortar

TED_meters = set(['jesse-turner-center'])

client = pymortar.Client()

def get_greenbutton_id(site, use_TED_meter=False):
    if use_TED_meter: 
        power_query = """SELECT ?meter WHERE {
            ?meter rdf:type/rdfs:subClassOf* brick:Building_Electric_Meter
        };"""
    else:
        power_query = """SELECT ?meter WHERE {
                ?meter rdf:type/rdfs:subClassOf* brick:Green_Button_Meter
            };"""
    query_agg = pymortar.MAX
    start = '2019-01-01T00:00:00-08:00'
    end = '2019-01-02T00:00:00-08:00'
    request = pymortar.FetchRequest(
        sites=[site],
        views = [
            pymortar.View(name='power', definition=power_query)
        ],
        time = pymortar.TimeParams(start=start, end=end),
        dataFrames=[
            pymortar.DataFrame(
            name='power',
            aggregation=query_agg,
            window='24h',
            timeseries=[
                pymortar.Timeseries(
                    view='power',
                    dataVars=['?meter'])
            ])
        ]
    )
    result = client.fetch(request)
    return result['power'].columns[0]
    

# print(get_greenbutton_id('ciee', "2018-01-01T10:00:00-07:00", "2018-08-12T10:00:00-07:00"))