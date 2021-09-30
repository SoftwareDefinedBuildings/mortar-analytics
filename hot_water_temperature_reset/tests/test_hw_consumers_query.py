import brickschema
import sys
import time
from os.path import join
import pandas as pd


sys.path.append('../')
import bacnet_and_data.readWriteProperty as BACpypesAPP
from bacnet_and_data.ControlledBoiler import ControlledBoiler as Boiler

# debugging setting
_debug = 1

if _debug: print("\nSetting up communication with BACnet network.\n")
BACnet_init_filename = join("../", "bacnet_and_data", "BACnet_init_temp_reset_2.ini")
access_bacnet = BACpypesAPP.Init(BACnet_init_filename)

# define brick schema, extension, and building model
schema_folder = join("../", "schema_and_models")

brick_schema_file = join(schema_folder, "Brick.ttl")
bldg_brick_model = join(schema_folder, "dbc.ttl")
brick_extensions = [
    join(schema_folder, "radiant_system_extension.ttl"),
    join(schema_folder, "bacnet_extension.ttl")
    ]


def _query_hw_consumers(g):
    """
    Retrieve hot water consumers in the building, their respective
    boiler(s), and relevant hvac zones.
    """
    # query direct and indirect hot water consumers
    hw_consumers_query = """SELECT DISTINCT * WHERE {
    ?boiler     rdf:type/rdfs:subClassOf?   brick:Boiler .
    ?boiler     brick:feeds+                ?t_unit .
    ?t_unit     rdf:type                    ?equip_type .
    ?mid_equip  brick:feeds                 ?t_unit .
    ?t_unit     brick:feeds+                ?room_space .
    ?room_space rdf:type/rdfs:subClassOf?   brick:HVAC_Zone .

        FILTER NOT EXISTS { 
            ?subtype ^a ?t_unit ;
                (rdfs:subClassOf|^owl:equivalentClass)* ?equip_type .
            filter ( ?subtype != ?equip_type )
            }
    }
    """
    if _debug: print("Retrieving hot water consumers for each boiler.\n")

    q_result = g.query(hw_consumers_query)
    df_hw_consumers = pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars])

    #df_unique_hw_consumers = df_hw_consumers.drop_duplicates(subset=["t_unit"])

    return df_hw_consumers


def _clean_metadata(df_hw_consumers):
    """
    Cleans metadata dataframe to have unique hot water consumers with
    most specific classes associated to other relevant information.
    """

    unique_t_units = df_hw_consumers.loc[:, "t_unit"].unique()
    direct_consumers_bool = df_hw_consumers.loc[:, 'mid_equip'] == df_hw_consumers.loc[:, 'boiler']

    direct_consumers = df_hw_consumers.loc[direct_consumers_bool, :]
    indirect_consumers = df_hw_consumers.loc[~direct_consumers_bool, :]

    # remove any direct hot consumers listed in indirect consumers
    for unit in direct_consumers.loc[:, "t_unit"].unique():
        indir_test = indirect_consumers.loc[:, "t_unit"] == unit

        # update indirect consumers df
        indirect_consumers = indirect_consumers.loc[~indir_test, :]

    # label type of hot water consumer
    direct_consumers.loc[:, "consumer_type"] = "direct"
    indirect_consumers.loc[:, "consumer_type"] = "indirect"

    hw_consumers = pd.concat([direct_consumers, indirect_consumers])
    hw_consumers = hw_consumers.drop(columns=["subtype"]).reset_index(drop=True)

    return hw_consumers



if __name__ == "__main__":

    # load schema files
    g = brickschema.Graph()

    if True:
        if _debug: print("Loading in building's Brick model.\n")
        g.load_file(brick_schema_file)
        [g.load_file(fext) for fext in brick_extensions]
        g.load_file(bldg_brick_model)

        # expand Brick graph
        if _debug: print("Expanding and inferring Brick graph.")
        print(f"Starting graph has {len(g)} triples")

        g.expand(profile="owlrl")

        print(f"Inferred graph has {len(g)} triples")

        # serialize inferred Brick to output
        with open("dbc_brick_expanded.ttl", "wb") as fp:
            fp.write(g.serialize(format="turtle").rstrip())
            fp.write(b"\n")
    else:
        if _debug: print("Loading existing, pre-expanded building Brick model.\n")
        expanded_brick_model = "dbc_brick_expanded.ttl"
        g.load_file(expanded_brick_model)

    if False:
        # validate Brick graph
        valid, _, resultsText = g.validate()
        if not valid:
            print("Graph is not valid!")
            print(resultsText)

            with open("debug-validation_results.txt", "w") as f:
                f.write(resultsText)
        else:
            print("VALID GRAPH!!")


    # query hot water consumers and clean metadata
    df_hw_consumers = _query_hw_consumers(g)
    df_hw_consumers = _clean_metadata(df_hw_consumers)

    # Define boilers to be controlled
    boilers2control = []
    for boiler in df_hw_consumers['boiler'].unique():
        boiler_consumers = df_hw_consumers.loc[:, "boiler"].isin([boiler])
        hw_consumers = df_hw_consumers.loc[boiler_consumers, :]
        boilers2control.append(Boiler(boiler, hw_consumers, g, BACpypesAPP))

    #boilers2control[0].run_test()
    import pdb; pdb.set_trace()
