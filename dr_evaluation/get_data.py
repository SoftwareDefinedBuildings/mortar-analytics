import pymortar
import pandas as pd

from .utils import get_closest_station

cli = pymortar.Client()

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
    result = cli.fetch(request)
    return result['power']

def get_df(site, start, end, agg='MEAN', interval='15min'):

    # Get weather
    weather = get_weather(site, start, end, agg=agg, window=interval, cli=cli)
    if weather.index.tz is None:
        weather.index = weather.index.tz_localize('UTC')
    weather.index = weather.index.tz_convert('US/Pacific')

    closest_station = get_closest_station(site)
    if closest_station is not None:
        weather = pd.DataFrame(weather[closest_station])
    else:
        weather = pd.DataFrame(weather.mean(axis=1))

    # Get power
    power = get_power(site, start, end, agg=agg, window=interval, cli=cli) * 4
    if power.index.tz is None:
        power.index = power.index.tz_localize('UTC')
    power.index = power.index.tz_convert('US/Pacific')

    # Merge
    power_sum = pd.DataFrame(power.sum(axis=1))
    data = power_sum.merge(weather, left_index=True, right_index=True)
    data.columns = ['power', 'weather']

    return data
