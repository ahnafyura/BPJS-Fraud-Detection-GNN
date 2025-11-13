# 🧠 GRAFANA: Fraud Graph Analytics with Neo4j + GDS

> **GRAFANA** (Graph Fraud Analytics) adalah implementasi analitik jaringan berbasis **Graph Database (Neo4j)** dan **Graph Data Science (GDS)** untuk mendeteksi potensi kolusi dan fraud pada klaim kesehatan.
> Proyek ini mengubah data tabular menjadi **struktur graf** yang menghubungkan pasien dan klaim, kemudian melakukan **analisis komunitas (Louvain)** untuk menemukan pola kolusif.

---

## ⚙️ 1. Persiapan Lingkungan

### ✅ Prasyarat

* **Neo4j Desktop** (versi 5.x)
* **Graph Data Science (GDS) plugin**
* **Python 3.10+** (opsional, jika melakukan ETL via script)

### 🔹 Langkah Awal

1. Jalankan **Neo4j Desktop**
2. Buat instance baru:

   * Name: `grafana_fraud_db`
   * Password: `neo4j123`
3. Start database → pastikan status **🟢 Running**
4. Pastikan plugin **Graph Data Science** sudah diinstall (tab *Plugins → Graph Data Science → Install*)
5. Di tab **Settings**, tambahkan konfigurasi berikut:

   ```ini
   dbms.security.procedures.unrestricted=gds.*
   dbms.security.procedures.allowlist=gds.*
   ```
6. Restart database

---

## 🧩 2. Cek Koneksi Database di Browser

Buka **Neo4j Browser** melalui
🔗 [http://localhost:7474](http://localhost:7474)

Login dengan:

```
Username: neo4j
Password: neo4j123
```

Lalu jalankan query sederhana untuk memastikan database aktif:

```cypher
RETURN "Neo4j is connected successfully!" AS status;
```

Output:

```
status
------------------------
Neo4j is connected successfully!
```

---

## 📢 3. Memuat Data Pasien dan Klaim

> Data berasal dari dataset dummy (contoh: `dummy_data.csv`) dengan struktur:
>
> | ID Klaim | ID Pasien | Diagnosis Utama | Prosedur | Tarif Seharusnya (Rp) | Tarif Diklaim (Rp) | Jenis Pelayanan | Lama Rawat (hari) | Kelas Rawat | Status Klaim | Jenis Fraud (Jika Terbukti) | Catatan | is_fraud |

Gunakan script Python berikut (`load_data.py`) untuk melakukan ETL dari CSV ke Neo4j:

```python
from py2neo import Graph, Node, Relationship
import pandas as pd

# 1. Koneksi ke Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "neo4j123"))

# 2. Load dataset
df = pd.read_csv("dummy_data.csv")
df.columns = df.columns.str.strip().str.lower()
print("Jumlah data:", len(df))

# 3. Bersihkan database
graph.run("MATCH (n) DETACH DELETE n;")

# 4. Masukkan node dan relasi
for _, row in df.iterrows():
    p = Node("Patient", id_pasien=row["id pasien"])
    c = Node(
        "Claim",
        id_klaim=row["id klaim"],
        diagnosis=row.get("diagnosis utama"),
        prosedur=row.get("prosedur"),
        tarif_seharusnya=row.get("tarif seharusnya (rp)"),
        tarif_diklaim=row.get("tarif diklaim (rp)"),
        jenis_pelayanan=row.get("jenis pelayanan"),
        lama_rawat=row.get("lama rawat (hari)"),
        kelas_rawat=row.get("kelas rawat"),
        status_klaim=row.get("status klaim"),
        jenis_fraud=row.get("jenis fraud (jika terbukti)"),
        catatan=row.get("catatan"),
        is_fraud=row.get("is_fraud")
    )

    graph.merge(p, "Patient", "id_pasien")
    graph.merge(c, "Claim", "id_klaim")
    graph.merge(Relationship(p, "MADE_CLAIM", c))

print("✅ Data klaim berhasil dimuat ke Neo4j!")
```

---

## 🧠 4. Membuat Proyeksi Graf untuk Analisis Louvain

Setelah data dimuat, jalankan query berikut di **Neo4j Browser**:

```cypher
// Membuat graph projection
CALL gds.graph.project(
  'fraud_graph',
  ['Patient', 'Claim'],
  {
    MADE_CLAIM: {orientation: 'UNDIRECTED'}
  }
);

// Jalankan algoritma Louvain untuk deteksi komunitas
CALL gds.louvain.write('fraud_graph', { writeProperty: 'community' });

// Lihat hasil komunitas
MATCH (n)
RETURN labels(n)[0] AS label, n.community AS community, COUNT(*) AS members
ORDER BY members DESC LIMIT 10;
```

📊 **Tujuan:**
Mengelompokkan node pasien–klaim berdasarkan tingkat konektivitas untuk mendeteksi potensi **kelompok kolusif** dalam sistem klaim.

---

## 🔍 5. Visualisasi Graf di Neo4j Browser

Untuk melihat jaringan klaim pasien:

```cypher
// Tampilkan sebagian graf (agar ringan)
MATCH (p:Patient)-[r:MADE_CLAIM]->(c:Claim)
RETURN p, r, c
LIMIT 50;
```

### 💡 Tips Visualisasi

1. Klik tab **Graph** di hasil query.
2. Klik ikon **⚙️ (gear)** → pilih **Color by property → community**

   > Setiap warna merepresentasikan komunitas berbeda hasil algoritma Louvain.
3. Zoom dan drag node untuk eksplorasi manual.

Contoh query untuk menampilkan klaim yang terindikasi fraud:

```cypher
MATCH (c:Claim {is_fraud: 1})<-[:MADE_CLAIM]-(p:Patient)
RETURN p, c;
```

---

## 🌈 6. (Opsional) Visualisasi dengan Neo4j Bloom

1. Buka **Neo4j Bloom** dari Neo4j Desktop (`Open → Neo4j Bloom`)
2. Gunakan perintah pencarian:

   ```
   MATCH (p:Patient)-[:MADE_CLAIM]->(c:Claim)
   RETURN p,c
   ```
3. Klik **Explore Scene**
4. Di panel kanan, atur warna node berdasarkan `community`
5. Export snapshot visual ke PNG untuk laporan riset

---

## 📄 7. Ekspor Hasil Analisis untuk GNN

> Jika ingin melatih **Graph Neural Network (GNN)** untuk prediksi fraud score per node.

Gunakan Python untuk mengekstrak data graf yang sudah diberi label `community`:

```python
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

df_nodes.to_csv("nodes.csv", index=False)
df_edges.to_csv("edges.csv", index=False)
print("📁 Data untuk GNN disimpan sebagai nodes.csv dan edges.csv")
```

---

## 🦯 8. Arsitektur Alur Proses

```
A[CSV Tabular Data] --> B[Python ETL Script]
B --> C[Neo4j Graph Database]
C --> D[Graph Data Science (Louvain)]
D --> E[Community Detection Result]
E --> F[Visualization in Browser / Bloom]
E --> G[GNN Training Dataset (nodes.csv, edges.csv)]
```

---

## 📘 9. Referensi

* Blondel, V. D., et al. (2008). *Fast unfolding of communities in large networks*.
  *Journal of Statistical Mechanics: Theory and Experiment*.
* Wang, Z., et al. (2025). *A robust and interpretable ensemble machine learning model for predicting healthcare insurance fraud.*
  *Scientific Reports, 15(1), 218.*

---

## 💬 10. Lisensi

Proyek ini dikembangkan untuk keperluan riset akademik dan pembuktian konsep.
Lisensi: **MIT License**
