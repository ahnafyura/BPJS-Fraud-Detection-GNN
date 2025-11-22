# Grafana: Integrasi Graph Database untuk Fraud Detection dengan Graph Neural Network & Algoritma Louvain

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

> **GRAFANA** (Graph Fraud Analytics) adalah sistem deteksi fraud cerdas yang menggabungkan kekuatan **Neo4j Graph Database** dengan arsitektur Deep Learning **GNN** dan klasifikasi komunitas **Louvain**.
>
> Sistem ini tidak hanya memetakan hubungan pasien-klaim, tetapi juga mempelajari pola struktural (embedding) untuk memprediksi anomali dengan akurasi tinggi, divisualisasikan langsung melalui **Neo4j Bloom**.
---

## 📑 **Table of Contents**

* [✨ Features](#-features)
* [🏗️ Architecture](#️-architecture--pipeline)
* [⚙️ Setup Environment](#️-setup-environment)
* [🔗 Graph Model Design](#-graph-model-design)
* [🌐 Visualizations](#-visualizations)
* [👁️ Graph Visualization](#️-graph-visualization)
* [📄 License](#-license)

---

## ✨ **Features**

<table align="center" width="100%" style="border: none; table-layout: fixed;">
<tr>
<td width="33%" align="center" style="padding: 20px;">
<h3>🔗 Knowledge Graph Construction</h3>
<img src="https://img.shields.io/badge/Neo4j-Graph_Modeling-00d9ff?style=for-the-badge&logo=neo4j" />
<p>Mengubah data tabular mentah menjadi graf cerdas yang menghubungkan entitas <b>Patient dan Claim</b> untuk mengungkap relasi tersembunyi.</p>
</td>
<td width="33%" align="center" style="padding: 20px;">
<h3>🧬 Structural Feature Engineering</h3>
<img src="https://img.shields.io/badge/Algo-Louvain_&_Node2Vec-4ecdc4?style=for-the-badge" />
<p>Mengekstraksi fitur graf tingkat lanjut menggunakan algoritma <b>Louvain Community Detection</b>  untuk menangkap konteks komunitas fraud.</p>
</td>
<td width="33%" align="center" style="padding: 20px;">
<h3>🤖 Hybrid AI Prediction</h3>
<img src="https://img.shields.io/badge/Model-GraphSAGE_+_GAT_+_XGBoost-f39c12?style=for-the-badge&logo=pytorch" />
<p>Model ensemble yang menggabungkan kekuatan induktif <b>GraphSAGE</b> untuk klasifikasi risiko tinggi.</p>
</td>
</tr>
</table>

---

## 🏗️ Architecture & Pipeline

```mermaid
flowchart TD
    %% Data Ingestion
    DATA[📄 Raw CSV Data] -->|ETL: load_data.py| NEO4J[(🍃 Neo4j Database)]

    %% Graph Data Science
    NEO4J -->|Graph Projection| GDS[⚙️ Neo4j GDS Library]
    GDS -->|Community Detection| LOUVAIN[Louvain Algorithm]
    GDS -->|Structural Embedding| N2V[Node2Vec]

    %% AI Modeling
    subgraph AI_Core [🧠 Hybrid AI Engine]
        LOUVAIN & N2V -->|Export Features| HYBRID[Hybrid GNN Model]
        HYBRID -->|GraphSAGE + GAT| EMBED[Node Embeddings]
        EMBED -->|Ensemble| XGB[XGBoost Classifier]
    end

    %% Output & Viz
    XGB -->|Risk Score & Explanation| RESULT[📄 Final Report CSV]
    RESULT -->|Write Back: update_db.py| NEO4J
    NEO4J -->|Visual Investigation| BLOOM[🌸 Neo4j Bloom]
```

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
git clone https://github.com/ahnafyura/GRAFANA
cd GRAFANA
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

## 📦 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Library inti:

* `pandas`
* `py2neo`
* `torch`
* `scikit-learn`
* `torch-geometric`
* `xgboost`
* `node2vec`
---

# 🏗️ Neo4j Setup

## 1. Instalasi Neo4j Desktop

Download: [https://neo4j.com/download/](https://neo4j.com/download/)

Setelah instalasi:

1. Buat database baru
2. Username & password default:
  * neo4j
  * neo4j123
3. Jalankan database

# 🚀 2. Quick Start

Run full ETL, Louvain, and GNN pipeline
```bash
./wrapper.sh
```

Run individual steps
```bash
python -m etl.load
python -m louvain.louvain
python -m etl.export
python -m gnn.hybrid_gnn
```

# 🔗 Graph Model Design

## Node Types

* **Claim**
* **Patient**

# 👁️ Graph Visualization

## Neo4j Browser

Melihat 50 Claim:

```cypher
MATCH (c:Claim)-[r]-(n)
RETURN * LIMIT 50;
```

---

## 📄 **License**

MIT License
