__author__ = "Anand Krishnan Prakash"
__email__ = "akprakash@lbl.gov"

import pymortar
import datetime
import pandas as pd
import argparse

def get_query():
    query = """SELECT ?point ?point_type WHERE { ?point rdf:type/rdfs:subClassOf* brick:Point . ?point rdf:type ?point_type . };"""
    return query

def get_all_points(client, site=None):
    query = get_query()

    resp = client.qualify([query])
    if resp.error != "" :
        print("ERROR: ", resp.error)

    if site!=None and site not in resp.sites:
        return pd.DataFrame()

    if len(resp.sites) == 0:
        return pd.DataFrame()

    if site !=None:
        sites = [site]
    else:
        sites = resp.sites

    points_view = pymortar.View(
        sites=sites,
        name="point_type_data",
        definition=query,
    )

    time_params = pymortar.TimeParams(
        start="2017-01-01T00:00:00Z",
        end="2018-01-01T00:00:00Z",
    )

    request = pymortar.FetchRequest(
        sites=sites,
        views=[points_view],
        time=time_params
    )

    response = client.fetch(request)
    if len(response.tables) == 0:
    	return pd.DataFrame()

    sites = [site[0] for site in response.query('select distinct site from point_type_data')]

    df_list = []
    for site in sites:
        q = """
                SELECT point, point_type
                FROM point_type_data
                WHERE site = "{0}";
            """.format(site)
        out = response.query(q)
        points = []
        point_types = []
        for point, point_type in out:
            point = point.split('#')[1]
            point_type = point_type.split('#')[1]
#           print(point+": "+point_type)
            points.append(point)
            point_types.append(point_type)
        df = pd.DataFrame(data={'point':points, 'type': point_types})
        df['site'] = site
        df_list.append(df)

    df = pd.concat(df_list)
    df = df.set_index('point')

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configure app parameters')
    parser.add_argument("-site", help="site whose points you want to find", type=str, default=None, nargs='?')
    
    site = parser.parse_args().site

    client = pymortar.Client({})

    points_df = get_all_points(client=client, site=site)
    if not points_df.empty:
        if site == None:
            filename = "points_all.csv"
        else:
            filename = "points_{0}.csv".format(site)
        
        print("writing to {0}".format(filename))
        points_df.to_csv(filename)
    else:
        print("No points found")