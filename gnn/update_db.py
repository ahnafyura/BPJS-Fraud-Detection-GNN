# update_db.py
import pandas as pd
from py2neo import Graph
from config import env

def run():
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
    SET n.fraud_certainty = row.fraud_certainty,
        n.fraud_label = toInteger(row.is_fraud),
        n.risk_category = CASE 
            WHEN row.fraud_certainty > 0.8 THEN 'High Risk'
            WHEN row.fraud_certainty > 0.5 THEN 'Medium Risk'
            ELSE 'Low Risk'
        END
    """

    # Konversi dataframe ke list of dicts untuk dikirim ke Neo4j
    data_batch = df[['node_id', 'fraud_certainty', 'is_fraud']].to_dict('records') # type: ignore

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
    WHERE n.fraud_certainty IS NOT NULL 
    RETURN id(n), labels(n)[0], n.fraud_certainty, n.risk_category 
    ORDER BY n.fraud_certainty DESC LIMIT 5
    """
    result = graph.run(check_query).to_data_frame()
    print(result)

if __name__ == "__main__":
    run()