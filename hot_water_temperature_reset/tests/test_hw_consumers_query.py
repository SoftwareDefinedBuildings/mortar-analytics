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


def return_class_entity(g, entity):
    """
    Return the most specific class for an entity
    """

    term_query = """ SELECT * WHERE {
        ?entity rdf:type/rdfs:subClassOf? ?entity_class

            FILTER NOT EXISTS {
                ?subtype ^a ?entity ;
                    (rdfs:subClassOf|^owl:equivalentClass)* ?entity_class .
                filter ( ?subtype != ?entity_class )
                }
    }
    """

    term_query_result = g.query(term_query, initBindings={"entity": entity})
    df_term_query_result = pd.DataFrame(term_query_result, columns=[str(s) for s in term_query_result.vars])

    entity_class = df_term_query_result["entity_class"].unique()

    if len(entity_class) > 1:
        entity_class = g.get_most_specific_class(entity_class)

    if len(entity_class) == 0:
        return None

    return entity_class[0]


def query_hw_consumer_valve(g, hw_consumer):
    """
    Retrieve control valves for hot water consumers
    """

    vlv_query = """SELECT DISTINCT * WHERE {
        VALUES ?ctrl_equip { brick:Position_Sensor brick:Valve_Command }
        ?point_name     rdf:type                        ?ctrl_equip .
        ?point_name     brick:isPointOf                 ?t_unit .
        ?point_name     brick:bacnetPoint               ?bacnet_id .
        ?point_name     brick:hasUnit?                  ?val_unit .
        ?bacnet_id      brick:hasBacnetDeviceInstance   ?bacnet_instance .
        ?bacnet_id      brick:hasBacnetDeviceType       ?bacnet_type .
        ?bacnet_id      brick:accessedAt                ?bacnet_net .
        ?bacnet_net     dbc:connstring                  ?bacnet_addr .

        FILTER NOT EXISTS {
            VALUES ?exclude_tags {tag:Reversing tag:Damper }
            ?point_name brick:hasTag ?exclude_tags.
        }
    }"""

    q_result = g.query(vlv_query, initBindings={"t_unit": hw_consumer})
    df = pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars])

    # Verify that unit is a qudt unit
    #TODO figure out why brick:hasUnit is returning other objects
    qudt_units = df["val_unit"].str.contains("qudt.org")
    if any(qudt_units):
        df = df.loc[qudt_units, :]
    else:
        df.loc[~qudt_units, "val_unit"] = None

    df = df.drop_duplicates(subset=['point_name']).reset_index(drop=True)

    if df.shape[0] == 0:
        return None

    return dict(df.loc[df.index[0]])


def query_hw_mode_status(g, hw_consumer):
    """
    Get mode status (heating or cooling) of hot water consumers
    """

    mode_query = """SELECT DISTINCT * WHERE {
        VALUES ?mode_status {
            brick:Heating_Start_Stop_Status brick:Heating_Enable_Command
            brick:Cooling_Start_Stop_Status brick:Cooling_Enable_Command
            }

        ?point_name     rdf:type                        ?mode_status .
        ?point_name     brick:isPointOf                 ?t_unit .
        ?point_name     brick:bacnetPoint               ?bacnet_id .
        ?point_name     brick:hasUnit?                  ?val_unit .
        ?bacnet_id      brick:hasBacnetDeviceInstance   ?bacnet_instance .
        ?bacnet_id      brick:hasBacnetDeviceType       ?bacnet_type .
        ?bacnet_id      brick:accessedAt                ?bacnet_net .
        ?bacnet_net     dbc:connstring                  ?bacnet_addr .
    }"""

    q_result = g.query(mode_query, initBindings={"t_unit": hw_consumer})
    df = pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars])

    # Verify that unit is a qudt unit
    #TODO figure out why brick:hasUnit is returning other objects
    qudt_units = df["val_unit"].str.contains("qudt.org")
    if any(qudt_units):
        df = df.loc[qudt_units, :]
    else:
        df.loc[~qudt_units, "val_unit"] = None

    status_entities = {
        "Heating_Start_Stop_Status": True,
        "Cooling_Start_Stop_Status": False,
        "Heating_Enable_Command": True,
        "Cooling_Enable_Command": False,
    }

    indicate_status = None
    if df.shape[0] > 1:
        for entity in status_entities:
            mode_found = df["mode_status"].str.contains(entity)
            if any(mode_found):
                df = df.loc[mode_found, :]
                if status_entities[entity]:
                    indicate_status = 'heating'
                else:
                    indicate_status = 'not heating'
    else:
        return (None, None)

    # for col in df.columns: print(col, df[col].unique(), '\n')

    return (dict(df.loc[df.index[0]]), indicate_status)



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

    # test ID of entity class
    entity_class = [return_class_entity(g, t_unit) for t_unit in df_hw_consumers["t_unit"]]
    df_hw_consumers.loc[:, "equip_type"] = entity_class

    # test ID of consumer's control valve
    control_vlvs = [query_hw_consumer_valve(g, t_unit) for t_unit in df_hw_consumers["t_unit"]]
    df_hw_consumers.loc[:, "ctrl_vlv"] = control_vlvs


    # test ID of consumer's mode status
    mode_status = [query_hw_mode_status(g, t_unit) for t_unit in df_hw_consumers["t_unit"]]
    df_hw_consumers.loc[:, "mode_status"] = [m[0] for m in mode_status]
    df_hw_consumers.loc[:, "mode_status_type"] = [m[1] for m in mode_status]


    # test ID of terminal response
    htm_terminals  = [
        "TABS_Panel", "ESS_Panel",
        "Thermally_Activated_Building_System_Panel",
        "Embedded_Surface_System_Panel",
        ]

    df_hw_consumers.loc[:, "htm"] = df_hw_consumers["equip_type"].str.contains('|'.join(htm_terminals))
    df_hw_consumers.loc[df_hw_consumers["htm"].isin([None]), "htm"] = False
    df_vlv = pd.DataFrame.from_records(list(df_hw_consumers.loc[df_hw_consumers["htm"], "ctrl_vlv"]))



    # Define boilers to be controlled
    boilers2control = []
    for boiler in df_hw_consumers['boiler'].unique():
        boiler_consumers = df_hw_consumers.loc[:, "boiler"].isin([boiler])
        hw_consumers = df_hw_consumers.loc[boiler_consumers, :]
        boilers2control.append(Boiler(boiler, hw_consumers, g, BACpypesAPP))

    #boilers2control[0].run_test()
    import pdb; pdb.set_trace()
