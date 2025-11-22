from py2neo import Graph, Node, Relationship
import pandas as pd
from config import env

def load_data():
    # 1. Koneksi ke Neo4j
    graph = Graph(env.url, auth=(env.uname, env.pw))

    # 2. Load dataset
    df = pd.read_csv(env.RAW_INPUT_DATA)
    df.columns = df.columns.str.strip().str.lower()
    print("Jumlah data:", len(df))

    # 3. Bersihkan database
    graph.run("MATCH (n) DETACH DELETE n;")

    # (Patient)-[:MADE_CLAIM]->(Claim)
    # (Claim)-[:HAS_DIAGNOSIS]->(Diagnosis)
    # (Claim)-[:HAS_PROCEDURE]->(Procedure)
    # (Claim)-[:HAS_SERVICE_TYPE]->(ServiceType)
    # (Claim)-[:IN_CLASS]->(CareClass)

    # 4. Masukkan node dan relasi
    for _, row in df.iterrows():
        # Patient
        p = Node("Patient", id_pasien=row["id pasien"])

        # Claim node (store labels + numeric values)
        c = Node(
            "Claim",
            id_klaim=row["id klaim"],
            tarif_seharusnya=row.get("tarif seharusnya (rp)"),
            tarif_diklaim=row.get("tarif diklaim (rp)"),
            lama_rawat=row.get("lama rawat (hari)"),
            is_fraud=row.get("is_fraud"),
            fraud_type=row.get("jenis fraud (jika terbukti)"),
            catatan=row.get("catatan"),
            status_klaim=row.get("status klaim")
        )

        # Convert only key categorical fields into nodes
        diag = Node("Diagnosis", name=row.get("diagnosis utama")) if row.get("diagnosis utama") else None
        proc = Node("Procedure", name=row.get("prosedur")) if row.get("prosedur") else None
        service = Node("ServiceType", name=row.get("jenis pelayanan")) if row.get("jenis pelayanan") else None
        kelas = Node("CareClass", name=row.get("kelas rawat")) if row.get("kelas rawat") else None

        # Merge patient + claim
        graph.merge(p, "Patient", "id_pasien")
        graph.merge(c, "Claim", "id_klaim")

        # Patient → Claim
        graph.merge(Relationship(p, "MADE_CLAIM", c))

        # Diagnosis
        if diag:
            graph.merge(diag, "Diagnosis", "name")
            graph.merge(Relationship(c, "HAS_DIAGNOSIS", diag))

        # Procedure
        if proc:
            graph.merge(proc, "Procedure", "name")
            graph.merge(Relationship(c, "HAS_PROCEDURE", proc))

        # ServiceType
        if service:
            graph.merge(service, "ServiceType", "name")
            graph.merge(Relationship(c, "HAS_SERVICE_TYPE", service))

        # CareClass
        if kelas:
            graph.merge(kelas, "CareClass", "name")
            graph.merge(Relationship(c, "IN_CLASS", kelas))
    print("Load data success")

if __name__ == "__main__":
    load_data()