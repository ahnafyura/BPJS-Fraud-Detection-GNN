# load_data.py
from py2neo import Graph, Node, Relationship
import pandas as pd

# === 1. Koneksi ke Neo4j ===
graph = Graph("bolt://localhost:7687", auth=("neo4j", "neo4j123"))

# === Ekspor hasil untuk training GNN ===

# Ambil semua node dan propertinya
q_nodes = """
MATCH (n)
RETURN id(n) AS node_id, 
       labels(n)[0] AS node_label,
       n.community AS community,
       n.is_fraud AS is_fraud,
       n.tarif_diklaim AS tarif_diklaim,
       n.tarif_seharusnya AS tarif_seharusnya,
       n.lama_rawat AS lama_rawat,
       n.kelas_rawat AS kelas_rawat,
       n.jenis_pelayanan AS jenis_pelayanan
"""
nodes = graph.run(q_nodes).to_data_frame()

# Ambil semua relasi antar node
q_edges = """
MATCH (a)-[r:MADE_CLAIM]->(b)
RETURN id(a) AS source, id(b) AS target, type(r) AS relation
"""
edges = graph.run(q_edges).to_data_frame()

# Simpan ke CSV
nodes.to_csv("nodes.csv", index=False)
edges.to_csv("edges.csv", index=False)

print("✅ Data diekspor: nodes.csv dan edges.csv")


# # === 2. Load dataset ===
# df = pd.read_csv("claims_synthetic_1200.csv")  # ganti path sesuai file kamu
# df.columns = df.columns.str.strip().str.lower()  # normalisasi kolom
# print("Kolom CSV:", df.columns.tolist())
# print("Jumlah data:", len(df))

# # === 3. Bersihkan database (opsional) ===
# graph.run("MATCH (n) DETACH DELETE n;")

# # === 4. Load data ke Neo4j ===
# for _, row in df.iterrows():
#     patient_id = row["id pasien"]
#     claim_id = row["id klaim"]

#     # Node Pasien
#     p = Node("Patient", id_pasien=patient_id)

#     # Node Klaim
#     c = Node(
#         "Claim",
#         id_klaim=claim_id,
#         diagnosis=row.get("diagnosis utama"),
#         prosedur=row.get("prosedur"),
#         tarif_seharusnya=row.get("tarif seharusnya (rp)"),
#         tarif_diklaim=row.get("tarif diklaim (rp)"),
#         jenis_pelayanan=row.get("jenis pelayanan"),
#         lama_rawat=row.get("lama rawat (hari)"),
#         kelas_rawat=row.get("kelas rawat"),
#         status_klaim=row.get("status klaim"),
#         jenis_fraud=row.get("jenis fraud (jika terbukti)"),
#         catatan=row.get("catatan"),
#         is_fraud=row.get("is_fraud")
#     )

#     # Masukkan node dan relasi
#     graph.merge(p, "Patient", "id_pasien")
#     graph.merge(c, "Claim", "id_klaim")
#     graph.merge(Relationship(p, "MADE_CLAIM", c))

# print("✅ Data klaim berhasil dimuat ke Neo4j!")

# # === 5. Verifikasi jumlah node ===
# count_nodes = graph.run("MATCH (n) RETURN COUNT(n) AS total;").data()
# print("Total node di Neo4j:", count_nodes)
