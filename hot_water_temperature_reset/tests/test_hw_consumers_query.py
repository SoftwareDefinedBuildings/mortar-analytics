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


def query_hw_consumers(g):
    """
    Retrieve hot water consumers in the building, their respective
    boiler(s), and relevant hvac zones.
    """
    # query direct and indirect hot water consumers
    hw_consumers_query = """ SELECT DISTINCT * WHERE {
        VALUES ?t_type { brick:Equipment brick:Water_Loop }
        ?boiler     rdf:type/rdfs:subClassOf?   brick:Boiler .
        ?boiler     brick:feeds+                ?t_unit .
        ?t_unit     brick:isFedBy               ?mid_equip .
        ?t_unit     rdf:type/rdfs:subClassOf?   ?t_type .
    }
    """

    if _debug: print("Retrieving hot water consumers for each boiler.\n")

    q_result = g.query(hw_consumers_query)
    df_hw_consumers = pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars])

    # Verify that intermediate equipment is part of hot water system
    boilers = list(df_hw_consumers["boiler"].unique())
    terminal_units = list(df_hw_consumers["t_unit"].unique())
    part_hw_sys = df_hw_consumers["mid_equip"].isin(boilers + terminal_units)

    #df_unique_hw_consumers = df_hw_consumers.drop_duplicates(subset=["t_unit"]
    return df_hw_consumers.loc[part_hw_sys, :]


def clean_metadata(df_hw_consumers):
    """
    Cleans metadata dataframe to have unique hot water consumers with
    most specific classes associated to other relevant information.
    """

    direct_consumers_bool = df_hw_consumers.loc[:, 'mid_equip'] == df_hw_consumers.loc[:, 'boiler']

    df_hw_consumers.loc[direct_consumers_bool, "consumer_type"] = "direct"
    df_hw_consumers.loc[~direct_consumers_bool, "consumer_type"] = "indirect"

    return df_hw_consumers



if __name__ == "__main__":

    # load schema files
    g = brickschema.Graph()

    expanded_brick_model = "../dbc_brick_expanded.ttl"

    if False:
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
        with open(expanded_brick_model, "wb") as fp:
            fp.write(g.serialize(format="turtle").rstrip())
            fp.write(b"\n")
    else:
        if _debug: print("Loading existing, pre-expanded building Brick model.\n")
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
    df_hw_consumers = query_hw_consumers(g)
    df_hw_consumers = clean_metadata(df_hw_consumers)

    # Define boilers to be controlled
    boilers2control = []
    for boiler in df_hw_consumers['boiler'].unique():
        boiler_consumers = df_hw_consumers.loc[:, "boiler"].isin([boiler])
        hw_consumers = df_hw_consumers.loc[boiler_consumers, :]
        boilers2control.append(Boiler(boiler, hw_consumers, g, BACpypesAPP))

    #boilers2control[0].run_test()
    import pdb; pdb.set_trace()
