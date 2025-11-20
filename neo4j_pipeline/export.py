import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py2neo import Graph, Node, Relationship
import pandas as pd
import neo4j_pipeline.config as config

# Koneksi ke Neo4j
graph = Graph(config.url, auth=(config.uname, config.pw))

# ===============================
# 1) EXPORT NODE FEATURES LENGKAP
# ===============================
q_nodes = """
MATCH (n)
RETURN
    id(n) AS node_id,
    labels(n)[0] AS label,

    // community detection features
    n.community AS community_id,
    n.community_size AS community_size,
    n.community_density AS community_density,

    // graph structural features
    n.degree AS degree,
    n.pagerank AS pagerank,
    n.betweenness AS betweenness,
    n.closeness AS closeness,

    // domain tabular features (Claim nodes only)
    n.tarif_seharusnya AS tarif_seharusnya,
    n.tarif_diklaim AS tarif_diklaim,
    n.lama_rawat AS lama_rawat,

    // fraud label (supervised)
    n.is_fraud AS fraud_label
"""
df_nodes = graph.run(q_nodes).to_data_frame()

# ===============================
# 2) EXPORT EDGES
# ===============================
q_edges = """
MATCH (a)-[r]->(b)
RETURN
    id(a) AS source,
    id(b) AS target,
    type(r) AS relation
"""
df_edges = graph.run(q_edges).to_data_frame()

# ===============================
# 3) SAVE TO /data/processed/
# ===============================
output_folder = "data/processed/"
os.makedirs(output_folder, exist_ok=True)

df_nodes.to_csv(os.path.join(output_folder, "neo4js_nodes.csv"), index=False)
df_edges.to_csv(os.path.join(output_folder, "neo4js_edges.csv"), index=False)

print("✅ Semua data graph berhasil diekspor ke data/processed/")
