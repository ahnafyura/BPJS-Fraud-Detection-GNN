# Grafana: Integrasi Graph Database untuk Fraud Detection dengan Graph Neural Networks & Algoritma Louvain

<div align="center">

<table style="border: none; margin: 0 auto; padding: 0; border-collapse: collapse;">
<tr>
<td align="center" style="vertical-align: middle; padding: 10px; border: none; width: 250px;">
  <img src="img/grafana_logo.png" alt="GRAFANA Logo" width="200"/>
</td>
<td align="left" style="vertical-align: middle; padding: 10px 0 10px 30px; border: none;">
  <pre style="font-family: 'Courier New', monospace; font-size: 16px; color: #0EA5E9; margin: 0; padding: 0; text-shadow: 0 0 10px #0EA5E9, 0 0 20px rgba(14,165,233,0.5); line-height: 1.2; transform: skew(-1deg, 0deg); display: block;">

░██████╗░██████╗░░█████╗░███████╗░█████╗░███╗░░██╗░█████╗░
██╔════╝░██╔══██╗██╔══██╗██╔════╝██╔══██╗████╗░██║██╔══██╗
██║░░██╗░██████╔╝███████║█████╗░░███████║██╔██╗██║███████║
██║░░╚██╗██╔══██╗██╔══██║██╔══╝░░██╔══██║██║╚████║██╔══██║
╚██████╔╝██║░░██║██║░░██║██║░░░░░██║░░██║██║░╚███║██║░░██║
░╚═════╝░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░░░░╚═╝░░╚═╝╚═╝░░╚══╝╚═╝░░╚═╝
  </pre>
</td>
</tr>
</table>

<p>
  <img src="https://img.shields.io/badge/Neo4j-GraphDB-00d9ff?style=for-the-badge&logo=neo4j&logoColor=white"/>
  <img src="https://img.shields.io/badge/GDS-Graph_Data_Science-4ecdc4?style=for-the-badge&logo=protodotio&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-ETL_Scripts-f39c12?style=for-the-badge&logo=python&logoColor=white"/>
</p>

<div align="center">
<a href="https://trendshift.io/repositories/14665" target="_blank"><img src="https://trendshift.io/api/badge/repositories/14665" alt="Grafana Team" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<div align="center" style="width: 100%; height: 2px; margin: 20px 0; background: linear-gradient(90deg, transparent, #00d9ff, transparent);"></div>
</div>

> **GRAFANA** (Graph Fraud Analytics) adalah framework analitik fraud berbasis **Neo4j Graph Database** dan **Graph Data Science (GDS)** untuk mendeteksi jaringan kolusi dalam klaim kesehatan.
> Sistem ini mengubah data tabular menjadi graf pasien–klaim dan menganalisis komunitas menggunakan algoritma **Louvain**.

---

## 📑 **Table of Contents**

* [✨ Features](#-features)
* [🏗️ Architecture](#️-architecture)
* [⚙️ Setup Environment](#️-setup-environment)
* [📥 Data Loading (ETL)](#-data-loading-etl)
* [🧠 Graph Projection + Louvain](#-graph-projection--louvain)
* [🌐 Visualizations](#-visualizations)
* [📁 Export for GNN](#-export-for-gnn)
* [📄 License](#-license)

---

## ✨ **Features**

<table align="center" width="100%" style="border: none; table-layout: fixed;">
<tr>
<td width="33%" align="center" style="padding: 20px;">
<h3>🔗 Graph-Based Fraud Mapping</h3>
<img src="https://img.shields.io/badge/Graph-Modeling-00d9ff?style=for-the-badge" />
<p>Mengonversi data tabular menjadi graf kompleks yang menghubungkan pasien, klaim, diagnosis, dan fasilitas kesehatan.</p>
</td>
<td width="33%" align="center" style="padding: 20px;">
<h3>🧠 Community Detection</h3>
<img src="https://img.shields.io/badge/GDS-Louvain-4ecdc4?style=for-the-badge" />
<p>Mendeteksi kelompok yang berpotensi melakukan kolusi berdasarkan struktur keterhubungan.</p>
</td>
<td width="33%" align="center" style="padding: 20px;">
<h3>📊 GNN Integration</h3>
<img src="https://img.shields.io/badge/GNN-Dataset_Export-f39c12?style=for-the-badge" />
<p>Mengekspor nodes & edges untuk pelatihan Graph Neural Network.</p>
</td>
</tr>
</table>

---

## 🏗️ **Architecture**

```
A[Tabular CSV Data]
  --> B[Python ETL]
  --> C[Neo4j Graph Database]
  --> D[Graph Projection (GDS)]
  --> E[Louvain Community Detection]
  --> F[Graph Visualization]
  --> G[GNN Dataset Export]
```

---

# ⚙️ Setup Environment

Panduan ini menjelaskan seluruh instalasi dari nol hingga siap menjalankan pipeline GRAFANA.

## 🧱 1. System Requirements

* Python ≥ 3.10
* Neo4j Desktop / Neo4j AuraDB
* CUDA (opsional, untuk training GNN)
* Pip & Virtualenv

---

## 🐍 2. Create Virtual Environment

```bash
git clone https://github.com/username/GRAFANA
cd GRAFANA
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

## 📦 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Library inti:

* `neo4j`
* `pandas`, `numpy`
* `networkx`
* `torch`, `pyg` (PyTorch Geometric)
* `matplotlib`

---

# 🏗️ 4. Neo4j Setup

## 4.1 Instalasi Neo4j Desktop

Download: [https://neo4j.com/download/](https://neo4j.com/download/)

Setelah instalasi:

1. Buat database baru
2. Gunakan password: `neo4j` (atau custom)
3. Jalankan database

## 4.2 Import Data

Gunakan file `etl/claims.csv`, `etl/providers.csv`, dll.

Contoh import (Neo4j Browser):

```cypher
LOAD CSV WITH HEADERS FROM 'file:///claims.csv' AS row
CREATE (:Claim {
    claim_id: row.claim_id,
    amount: toFloat(row.amount),
    date: row.date
});
```

---

# 🔗 5. Graph Model Design

## Node Types

* **Claim**
* **Patient**
* **Provider**
* **Hospital**

## Relationship Types

* `(:Patient)-[:SUBMITTED]->(:Claim)`
* `(:Provider)-[:HANDLED]->(:Claim)`
* `(:Provider)-[:WORKS_AT]->(:Hospital)`

Diagram:

```
Patient ---SUBMITTED---> Claim <---HANDLED--- Provider ---WORKS_AT---> Hospital
```

---

# 🔄 6. ETL Pipeline

File: `etl/extract_to_neo4j.py`

### 6.1 Extract

```python
import pandas as pd
claims = pd.read_csv('data/claims.csv')
```

### 6.2 Transform

```python
claims['amount_norm'] = (claims['amount'] - claims['amount'].mean()) / claims['amount'].std()
```

### 6.3 Load to Neo4j

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(URI, auth=(USER, PASS))
```

---

# 👁️ 7. Graph Visualization

## 7.1 Neo4j Browser

Gunakan:

```cypher
MATCH (c:Claim)-[r]-(n)
RETURN * LIMIT 50;
```

## 7.2 Python Visualization

```python
import networkx as nx
import matplotlib.pyplot as plt
```

---

# 🧠 8. GNN Training

Menggunakan PyTorch Geometric.

## 8.1 Convert Neo4j → PyG

File: `gnn/neo4j_to_pyg.py`

Pipeline:

1. Query nodes & relationships
2. Encode categorical entities
3. Build `edge_index`
4. Build `node_features`

## 8.2 Train Model

File: `gnn/train.py`

Model: GraphSAGE / GAT

```python
model = GraphSAGE(hidden_channels=64)
```

## 8.3 Evaluate

```python
accuracy, f1 = evaluate(model, loader)
```

---

# 📁 9. Project Structure

```
GRAFANA/
│── etl/
│   ├── extract_to_neo4j.py
│   ├── claims.csv
│   └── providers.csv
│
│── gnn/
│   ├── neo4j_to_pyg.py
│   ├── train.py
│   └── model.py
│
│── assets/
│── README.md
│── requirements.txt
```

---

# 🚀 10. Quick Start

```bash
python etl/extract_to_neo4j.py
python gnn/neo4j_to_pyg.py
python gnn/train.py
```

## 📄 **License**

MIT License
