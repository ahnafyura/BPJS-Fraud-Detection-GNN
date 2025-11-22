# retrieve_data.py
from py2neo import Graph
import pandas as pd
from config import env

def retrieve_data_to_csv():
    # 1. Koneksi ke Neo4j
    graph = Graph(env.url, auth=(env.uname, env.pw))
    
    # 2. Query untuk mengambil semua data dengan fraud_certainty
    query = """
    MATCH (p:Patient)-[:MADE_CLAIM]->(c:Claim)
    OPTIONAL MATCH (c)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
    OPTIONAL MATCH (c)-[:HAS_PROCEDURE]->(proc:Procedure)
    OPTIONAL MATCH (c)-[:HAS_SERVICE_TYPE]->(st:ServiceType)
    OPTIONAL MATCH (c)-[:IN_CLASS]->(cc:CareClass)
    RETURN 
        c.id_klaim as `ID Klaim`,
        p.id_pasien as `ID Pasien`,
        d.name as `Diagnosis Utama`,
        proc.name as `Prosedur`,
        c.tarif_seharusnya as `Tarif Seharusnya (Rp)`,
        c.tarif_diklaim as `Tarif Diklaim (Rp)`,
        st.name as `Jenis Pelayanan`,
        c.lama_rawat as `Lama Rawat (hari)`,
        cc.name as `Kelas Rawat`,
        c.status_klaim as `Status Klaim`,
        c.fraud_type as `Jenis Fraud (Jika Terbukti)`,
        c.catatan as `Catatan`,
        c.is_fraud as `is_fraud`,
        c.fraud_certainty as `fraud_certainty`
    ORDER BY c.id_klaim
    """
    
    print("Retrieving data from Neo4j...")
    result = graph.run(query)
    df = result.to_data_frame()
    
    # 3. Add 'No' column as sequential number
    df.insert(0, 'No', range(1, len(df) + 1)) # type: ignore
    
    # 4. Handle missing fraud_certainty values (set to 0 if not exists)
    if 'fraud_certainty' not in df.columns:
        df['fraud_certainty'] = 0.0
    else:
        df['fraud_certainty'] = df['fraud_certainty'].fillna(0.0)
    
    # 5. Save to CSV
    output_file = env.FINAL_REPORT_CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"FINAL REPORT saved to {output_file}")
    print(f"Total records: {len(df)}")
    print("\nFirst 5 records:")
    print(df.head().to_string())
    
    return df

if __name__ == "__main__":
    retrieve_data_to_csv()