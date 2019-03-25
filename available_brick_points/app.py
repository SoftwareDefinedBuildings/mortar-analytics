__author__ = "Anand Krishnan Prakash"
__email__ = "akprakash@lbl.gov"

import pymortar
import datetime
import pandas as pd
import argparse

def get_all_points(client, site=None):
    query = """SELECT ?point ?point_type WHERE { ?point rdf:type/rdfs:subClassOf* brick:Point . ?point rdf:type ?point_type . };"""

    if site == None:
        resp = client.qualify([query])
        if resp.error != "" :
            print("ERROR: ", resp.error)
            return pd.DataFrame()
            
        if len(resp.sites) == 0:
            return pd.DataFrame()
        
        sites = resp.sites
    else:
        sites = [site]

    points_view = pymortar.View(
        sites=sites,
        name="point_type_data",
        definition=query,
    )

    request = pymortar.FetchRequest(
        sites=sites,
        views=[points_view]
    )

    response = client.fetch(request)
        
    if len(response.tables) == 0:
        return pd.DataFrame()

    view_df = response.view("point_type_data")
    view_df = view_df.rename({"point_type": "type"}, axis="columns")
    view_df = view_df.set_index('point')
    return view_df


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