"""
config.template.py

👉 File template untuk konfigurasi pribadi.
👉 Copy sebagai `config.py` dan masukkan kredensial asli (JANGAN commit ke git).
"""

# ======================
# NEO4J CONNECTION CONFIG
# ======================

# Contoh local Neo4j
url = "bolt://localhost:7687"

# Ganti dengan username/password Anda
uname = "neo4j"
pw = "YOUR_PASSWORD_HERE"

# Jika menggunakan Neo4j Aura (cloud):
url = "bolt://localhost:7687"
username = "neo4j"
password = "neo4j123"

# ======================
# OPTIONAL: PATH CONFIG
# ======================
# Lokasi default dataset
DATA_PATH = "data/raw/claims_synthetic_1200.csv"

# Lokasi processed file dari Neo4j export
NODES_PROCESSED = "data/processed/nodes_mapped.csv"
EDGES_PROCESSED = "data/processed/edges_mapped.csv"

# Lokasi hasil GNN
GNN_OUTPUT = "gnn_model/output/gnn_hybrid_predicted_nodes.csv"
