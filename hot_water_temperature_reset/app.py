import brickschema
from os.path import join
import pandas as pd


# define brick schema, extension, and building model
schema_folder = join('./', 'schema_and_models')

brick_schema_file = join(schema_folder, 'Brick.ttl')
bldg_brick_model = join(schema_folder, 'dbc.ttl')
brick_extensions = [
    join(schema_folder, 'radiant_system_extension.ttl'),
    join(schema_folder, 'bacnet_extension.ttl')
    ]

# load schema files
g = brickschema.Graph()
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
df_unique_hw_consumers = df_hw_consumers.drop_duplicates(subset=['t_unit'])
for index, row in df_unique_hw_consumers.iterrows():
    print(row['t_unit'])

import pdb; pdb.set_trace()

# query consumers' HVAC zone temperature and respective setpoints
for consumer in hw_consumers:
    q_result = g.query(consumer)
    for row in q_result:
        import pdb; pdb.set_trace()
        q_temp = f"""SELECT ?zn_t WHERE {{

        }}"""



import pdb; pdb.set_trace()

