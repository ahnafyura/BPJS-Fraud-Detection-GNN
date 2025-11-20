from py2neo import Graph, Node, Relationship
import pandas as pd
import neo4j_pipeline.config as config
import os

graph = Graph(config.url, auth=(config.uname, config.pw))

q_nodes = """
MATCH (n)
RETURN id(n) AS node_id, labels(n)[0] AS label, n.community AS community, n.is_fraud AS fraud_label
"""
df_nodes = graph.run(q_nodes).to_data_frame()

q_edges = """
MATCH (a)-[r]->(b)
RETURN id(a) AS source, id(b) AS target, type(r) AS relation
"""
df_edges = graph.run(q_edges).to_data_frame()

output_folder = ".out/"
os.makedirs(os.path.dirname(output_folder), exist_ok=True)

df_nodes.to_csv(os.path.join(output_folder, "nodes.csv"), index=False)
df_edges.to_csv(os.path.join(output_folder, "edges.csv"), index=False)
print("📁 Data untuk GNN disimpan sebagai nodes.csv dan edges.csv")