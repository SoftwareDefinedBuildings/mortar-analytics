__author__ = "Anand Krishnan Prakash"
__email__ = "akprakash@lbl.gov"

import pymortar
import datetime
import pandas as pd
import argparse


def get_error_message(x):
    dt_format = "%Y-%m-%d %H:%M:%S"
    st = x.name
    st_str = st.strftime(dt_format)
    site = x.site
    ahu = x.ahu
    msg = "At time: {0}, in the site: {1}, the AHU: {2} has both heating and cooling valves open".format(
            st_str,
            site,
            ahu
        )
    return msg


def ahu_analysis(client, start_time, end_time):
    st = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    et = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    query = """SELECT ?cooling_point ?heating_point ?ahu WHERE {
        ?cooling_point rdf:type/rdfs:subClassOf* brick:Cooling_Valve_Command .
        ?heating_point rdf:type/rdfs:subClassOf* brick:Heating_Valve_Command .
        ?ahu bf:hasPoint ?cooling_point .
        ?ahu bf:hasPoint ?heating_point .
    };"""

    resp = client.qualify([query])
    if resp.error != "":
        print("ERROR: ", resp.error)

    points_view = pymortar.View(
        sites=resp.sites,
        name="point_type_data",
        definition=query,
    )

    point_streams = pymortar.DataFrame(
        name="points_data",
        aggregation=pymortar.MAX,
        window="15m",
        timeseries=[
            pymortar.Timeseries(
                view="point_type_data",
                dataVars=["?cooling_point", "?heating_point"]
            )
        ]
    )

    time_params = pymortar.TimeParams(
        start=st,
        end=et
    )

    request = pymortar.FetchRequest(
        sites=resp.sites,
        views=[points_view],
        time=time_params,
        dataFrames=[
            point_streams
        ],
    )

    response = client.fetch(request)

    ahu_df = response["points_data"]
    ahus = [ahu[0] for ahu in response.query("select ahu from point_type_data")]


    error_df_list = []
    for ahu in ahus:
        heat_cool_query = """
            SELECT cooling_point_uuid, heating_point_uuid, site
            FROM point_type_data
            WHERE ahu = "{0}";
        """.format(ahu)
        res = response.query(heat_cool_query)
        cooling_uuid = res[0][0]
        heating_uuid = res[0][1]
        site = res[0][2]
        df = response["points_data"][[cooling_uuid, heating_uuid]].dropna()
        df.columns = ['cooling', 'heating']
        df['site'] = site
        df['ahu'] = ahu.split('#')[1]
        df['simultaneous_heat_cool'] = False
        df.loc[((df.cooling > 0) & (df.heating > 0)), 'simultaneous_heat_cool'] = True
        if not df[df['simultaneous_heat_cool'] == True].empty:
            error_df_list.append(df[df['simultaneous_heat_cool'] == True])

    if len(error_df_list) > 0:
        error_df = pd.concat(error_df_list, axis=0)[['site', 'ahu']]
        error_df.index.name = 'time'
        error_msgs = error_df.apply(lambda x: get_error_message(x), axis=1).values
        for msg in error_msgs:
            print(msg)

        return error_df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configure app parameters')
    # parser.add_argument("-time_interval", help="length of time interval (in minutes) when you want to check if a zone is both heating and cooling", type=int, default=60, nargs='?')
    parser.add_argument("-st", help="start time for analysis in yyyy-mm-ddThh:mm:ss format", type=str, default="2017-06-21T00:00:00", nargs='?')
    parser.add_argument("-et", help="end time for analysis in yyyy-mm-ddThh:mm:ss format", type=str, default="2017-07-01T00:00:00", nargs='?')
    parser.add_argument("-filename", help="filename to store result of analysis", type=str, default="simultaneous_heat_cool_ahu.csv", nargs='?')
    
    # resample_minutes = parser.parse_args().time_interval
    try:
        start_time =  datetime.datetime.strptime(parser.parse_args().st, "%Y-%m-%dT%H:%M:%S")
        end_time = datetime.datetime.strptime(parser.parse_args().et, "%Y-%m-%dT%H:%M:%S")
    except Exception as e:
        raise Exception("Incorrect format for st or et. Use yyyy-mm-ddThh:mm:ss")
    filename = parser.parse_args().filename

    client = pymortar.Client({})

    error_df = ahu_analysis(client=client, start_time=start_time, end_time=end_time)
    if not error_df.empty:
        print("Writing results to {0}".format(filename))
        error_df.to_csv(filename)
    else:
        print("No ahus match the condition")