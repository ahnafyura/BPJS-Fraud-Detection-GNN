# update_db.py
import pandas as pd
from py2neo import Graph
from config import env;

# 1. Konfigurasi
graph = Graph(env.url, auth=(env.uname, env.pw)) 

# File hasil training GNN terakhir
CSV_PATH = env.RETRAINED_OUTPUT_FILE

print("Reading prediction results file...")
df = pd.read_csv(CSV_PATH)

print(f"Updating {len(df)} nodes in Neo4j...")

# 2. Proses Update (Batching agar cepat)
# Kita akan menggunakan Cypher query dengan UNWIND untuk kecepatan tinggi
query = """
UNWIND $batch AS row
MATCH (n) WHERE id(n) = row.node_id
SET n.gnn_score = row.new_gnn_score,
    n.fraud_label = toInteger(row.is_fraud),
    n.risk_category = CASE 
        WHEN row.new_gnn_score > 0.8 THEN 'High Risk'
        WHEN row.new_gnn_score > 0.5 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END
"""

# Konversi dataframe ke list of dicts untuk dikirim ke Neo4j
data_batch = df[['node_id', 'new_gnn_score', 'is_fraud']].to_dict('records') # type: ignore

# Eksekusi Query
try:
    graph.run(query, batch=data_batch)
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")

# 3. Verifikasi
print("\nData Verification (Top 5 High Risk):")
check_query = """
MATCH (n) 
WHERE n.gnn_score IS NOT NULL 
RETURN id(n), labels(n)[0], n.gnn_score, n.risk_category 
ORDER BY n.gnn_score DESC LIMIT 5
"""
result = graph.run(check_query).to_data_frame()
print(result)