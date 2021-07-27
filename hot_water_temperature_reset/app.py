import brickschema
import sys
from os.path import join
import pandas as pd

# BACnet libraries
# sys.path.append("/etc/Github/brick-field_study/bacnet_explore/")
# import BACpypes_applications as BACpypesAPP

import readWriteProperty as BACpypesAPP

BACnet_init_filename = 'BACnet_init_temp_reset.ini'
BACpypesAPP.Init(BACnet_init_filename)

# define brick schema, extension, and building model
schema_folder = join("./", "schema_and_models")

brick_schema_file = join(schema_folder, "Brick.ttl")
bldg_brick_model = join(schema_folder, "dbc.ttl")
brick_extensions = [
    join(schema_folder, "radiant_system_extension.ttl"),
    join(schema_folder, "bacnet_extension.ttl")
    ]

def return_equipment_points(brick_point_class, brick_equipment_class):
    """
    Return defined brick point class for piece of equipment
    """
    # query to return all setpoints of equipment
    term_query = f"""SELECT DISTINCT ?bacnet_id ?t_unit ?t_unit_point ?point_type  WHERE {{
        ?t_unit         brick:hasPoint              ?t_unit_point .
        ?t_unit_point   rdf:type/rdfs:subClassOf*   brick:{brick_point_class} .
        ?t_unit_point   brick:bacnetPoint           ?bacnet_id .
        ?t_unit_point   rdf:type                    ?point_type .
        }}"""

    # execute the query
    term_query_result = g.query(term_query, initBindings={"t_unit": brick_equipment_class})

    df_term_query_result = pd.DataFrame(term_query_result, columns=[str(s) for s in term_query_result.vars])
    df_term_query_result["short_point_type"] = [shpt.split("#")[1] for shpt in df_term_query_result["point_type"]]
    df_unique_stpt_class = df_term_query_result.drop_duplicates(subset=["t_unit_point", "point_type"])

    return df_unique_stpt_class


def return_bacnet_point(df_term_query_result, point_priority):
    """
    Return bacnet point information for defined point class
    """

    # return bacnet point id for setpoint class that was found
    bacnet_id = None
    for stpt_class in point_priority:
        df_setpoint_class = df_term_query_result.loc[df_term_query_result["short_point_type"].isin([stpt_class]), :]
        if df_setpoint_class.shape[0] > 0:
            bacnet_id = df_setpoint_class.loc[df_setpoint_class.index[0], "bacnet_id"]
            break

    if bacnet_id is not None:
        bacnet_query = f"""SELECT ?bacnet_id ?t_unit_point ?bacnet_instance ?bacnet_type ?bacnet_addr ?point_type WHERE {{
            ?t_unit_point  brick:bacnetPoint                ?bacnet_id .
            ?bacnet_id     brick:hasBacnetDeviceInstance    ?bacnet_instance .
            ?bacnet_id     brick:hasBacnetDeviceType        ?bacnet_type .
            ?bacnet_id     brick:accessedAt                 ?bacnet_net .
            ?bacnet_net    dbc:connstring                   ?bacnet_addr .
            ?t_unit_point  rdf:type                         ?point_type .
            }}"""

        bacnet_query_result = g.query(bacnet_query, initBindings={"bacnet_id": bacnet_id})
        df_bacnet_query_result = pd.DataFrame(bacnet_query_result, columns=[str(s) for s in bacnet_query_result.vars]).drop_duplicates(subset=["bacnet_id"])
        df_bacnet_query_result["short_point_type"] = [shpt.split("#")[1] for shpt in df_bacnet_query_result["point_type"]]

    else:
        print("NO BACNET POINT ID FOUND!!")
        df_bacnet_query_result = None

    return df_bacnet_query_result


def return_equipment_controlled_temp_bacnet_point(zn_t_unit_name):
    """
    Return the bacnet id of the controlled temperature of the defined equipment
    """
    point_priority = [
        "Discharge_Air_Temperature_Sensor",
        "Embedded_Temperature_Sensor",
        "Hot_Water_Supply_Temperature_Sensor",
        "Discharge_Water_Temperature_Sensor",
        "Air_Temperature_Sensor",
        "Water_Temperature_Sensor",
        "Temperature_Sensor"
    ]

    brick_point_class = "Temperature_Sensor"
    brick_equipment_class = zn_t_unit_name

    df_term_query_result = return_equipment_points(brick_point_class, brick_equipment_class)

    sensor_bacnet_point = return_bacnet_point(df_term_query_result, point_priority)

    return(sensor_bacnet_point)


def return_equipment_setpoint_bacnet_point(hvac_mode, zn_t_unit_name):
    """
    Return setpoint for defined equipment
    """
    # define order of setpoint classes
    setpoint_priority = [
        f"Effective_Air_Temperature_{hvac_mode}_Setpoint",
        "Effective_Air_Temperature_Setpoint",
        f"Discharge_Air_Temperature_{hvac_mode}_Setpoint",
        f"Discharge_Air_Temperature_Setpoint",
        "Supply_Hot_Water_Temperature_Setpoint",
        "Discharge_Water_Temperature_Setpoint",
        f"{hvac_mode}_Temperature_Setpoint",
        "Air_Temperature_Setpoint",
        "Embedded_Temperature_Setpoint",
        "Water_Temperature_Setpoint"
        ]

    brick_point_class = f"Temperature_Setpoint"
    brick_equipment_class = zn_t_unit_name

    df_term_query_result = return_equipment_points(brick_point_class, brick_equipment_class)

    setpoint_bacnet_point = return_bacnet_point(df_term_query_result, setpoint_priority)

    return(setpoint_bacnet_point)


def bacnet_read(bacnet_point, read_attr='presentValue'):
    address  = str(bacnet_point['bacnet_addr'][0])
    obj_type = str(bacnet_point['bacnet_type'][0])
    obj_inst = int(bacnet_point['bacnet_instance'][0])

    args = [address, obj_type, obj_inst, read_attr]
    value_read = BACpypesAPP.read_prop(args)

    return value_read


def print_bacnet_point(bacnet_point, inside_bacnet=False, read_attr='presentValue'):
    """
    Print point name and info or read value off the bacnet network
    """
    if bacnet_point is not None:
        class_name = bacnet_point['short_point_type'][0]
        point_name = bacnet_point['t_unit_point'][0].split('#')[1]
        bacnet_id = bacnet_point['bacnet_instance'][0]
    else:
        print("BACnet point not found in brick model!")
        return None

    if not inside_bacnet:
        print(f"{point_name} ({class_name}) has BACnet ID of {bacnet_id}")
    else:
        value_read = bacnet_read(bacnet_point, read_attr=read_attr)
        print(f"Reading {point_name} ({class_name}) = {value_read}")



# load schema files
g = brickschema.Graph()

if False:
    g.load_file(brick_schema_file)
    [g.load_file(fext) for fext in brick_extensions]
    g.load_file(bldg_brick_model)

    # expand Brick graph
    print(f"Starting graph has {len(g)} triples")

    g.expand(profile="owlrl")

    print(f"Inferred graph has {len(g)} triples")

    # serialize inferred Brick to output
    with open("dbc_brick_expanded.ttl", "wb") as fp:
        fp.write(g.serialize(format="turtle").rstrip())
        fp.write(b"\n")
else:
    expanded_brick_model = "dbc_brick_expanded.ttl"
    g.load_file(expanded_brick_model)

# import pdb; pdb.set_trace()

# # validate Brick graph
# valid, _, resultsText = g.validate()
# if not valid:
#     print("Graph is not valid!")
#     print(resultsText)

#     with open("debug-validation_results.txt", "w") as f:
#         f.write(resultsText)
# else:
#     print("VALID GRAPH!!")


# query direct and indirect hot water consumers
hw_direct_consumers = """SELECT * WHERE {
  ?boiler     rdf:type/rdfs:subClassOf?   brick:Boiler .
  ?boiler     brick:feeds                 ?t_unit .
  ?t_unit     rdf:type                    ?equip_type .
}
"""

hw_indirect_consumers = """ SELECT * WHERE {
    ?boiler     rdf:type/rdfs:subClassOf?   brick:Boiler .
    ?boiler     brick:feeds                 ?equip .
    ?equip      brick:feeds                 ?t_unit .
    ?t_unit     rdf:type/rdfs:subClassOf?   brick:Terminal_Unit .
    ?t_unit     rdf:type                    ?equip_type .
}"""

hw_consumers = [hw_direct_consumers, hw_indirect_consumers]

df_container = []
for consumer in hw_consumers:
    q_result = g.query(consumer)
    df_container.append(pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars]))

df_hw_consumers = pd.concat(df_container, ignore_index=True, sort=False)
df_unique_hw_consumers = df_hw_consumers.drop_duplicates(subset=["t_unit"])

# read boiler equipement supply and setpoint temperatures
for boiler in df_hw_consumers['boiler'].unique():
    boiler_ctrl_temp_bacnet_point = return_equipment_controlled_temp_bacnet_point(boiler)
    boiler_stpt_bacnet_point = return_equipment_setpoint_bacnet_point('Heating', boiler)

    print_bacnet_point(boiler_ctrl_temp_bacnet_point, inside_bacnet=True, read_attr='presentValue')
    print_bacnet_point(boiler_stpt_bacnet_point, inside_bacnet=True, read_attr='presentValue')


# iterate through each equipment setpoints and zone temperatures
for i, equip_row in df_unique_hw_consumers.iterrows():
    zn_t_unit_name = equip_row["t_unit"]
    zn_t_unit_type = equip_row["equip_type"]
    print(f"\n{zn_t_unit_name}")

    hvac_mode = "Heating"

    equip_ctrl_temp_bacnet_point = return_equipment_controlled_temp_bacnet_point(zn_t_unit_name)
    equip_stpt_bacnet_point = return_equipment_setpoint_bacnet_point(hvac_mode, zn_t_unit_name)

    print_bacnet_point(equip_ctrl_temp_bacnet_point, inside_bacnet=True, read_attr='presentValue')
    print_bacnet_point(equip_stpt_bacnet_point, inside_bacnet=True, read_attr='presentValue')


import pdb; pdb.set_trace()