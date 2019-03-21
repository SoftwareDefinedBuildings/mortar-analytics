import pymortar

def get_weather(site, start, end, agg, window, cli):
    weather_query = """SELECT ?t WHERE {
            ?t rdf:type/rdfs:subClassOf* brick:Weather_Temperature_Sensor
        };"""
    query_agg = eval('pymortar.' + str.upper(agg))
    request = pymortar.FetchRequest(
        sites=[site],
        views = [
            pymortar.View(name='weather', definition=weather_query)
        ],
        time = pymortar.TimeParams(start=start, end=end),
        dataFrames=[
            pymortar.DataFrame(
            name='weather',
            aggregation=query_agg,
            window=window,
            timeseries=[
                pymortar.Timeseries(
                    view='weather',
                    dataVars=['?t'])
            ])
        ]
    )
    result = cli.fetch(request)
    return result['weather']

def get_power(site, start, end, agg, window, cli):
    power_query = """SELECT ?meter WHERE {
            ?meter rdf:type brick:Green_Button_Meter
        };"""
    query_agg = eval('pymortar.' + str.upper(agg))
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
            window=window,
            timeseries=[
                pymortar.Timeseries(
                    view='power',
                    dataVars=['?meter'])
            ])
        ]
    )
    result = client.fetch(request)
    return result['power']
