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

<table>
  <tr>
    <td width="50%">
      <h3 align="center"> Visualisasi Graf Pada Neo4j </h3>
      <div align="center">
        <img src="img/display.png" alt="Fraud Model" width="100%">
      </div>
      <p align="center">
        Memperlihatkan hubungan antar entitas dalam visualisasi graf <br>
        🔵 Biru (Claim)<br>
        🟤 Krem / Cokelat Muda (Patient) <br>
        🟠 Oranye (Diagnosis) <br>
        🟣 Ungu (Procedure) <br>
        🟢 Hijau (CareClass & ServiceType)
      </p>
    </td>
  </tr>
</table>

---

## 📑 **Table of Contents**

* [✨ Features](#-features)
* [🏗️ Architecture](#️-architecture--pipeline)
* [📊 Neo4j Bloom Visualization Results](#️-Neo4j-Bloom-Visualization-Results)
* [🖥️ Interface Showcase](#️-interface-showcase)
* [🧱 System Requirements](#-system-requirements)
* [🐍 Create Virtual Environment](#-create-virtual-environment)
* [📦 Install Dependencies](#-install-dependencies)
* [🏗️ Neo4j Setup](#️-neo4j-setup)
* [🚀 Quick Start](#-quick-start)
* [🔗 Graph Model Design](#-graph-model-design)
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
flowchart TB

    %% === DATA INGESTION ===
    DATA[Raw Claims CSV] -->|ETL Load| NEO4J[(Neo4j Database)]

    %% === GRAPH DATA SCIENCE ===
    NEO4J -->|Graph Projection| GDS[Neo4j GDS Engine]
    GDS -->|Louvain| LOUVAIN[Louvain Community Detection]
    GDS -->|Node2Vec| N2V[Node2Vec Embedding]

    %% === HYBRID AI ENGINE ===
    subgraph AI_Core [Hybrid AI Engine]
        LOUVAIN --> HYBRID[Hybrid GNN Model]
        N2V --> HYBRID
        HYBRID --> EMBED[GraphSAGE/GAT Embeddings]
        EMBED --> XGB[XGBoost Classifier]
    end

    %% === MERGE PIPELINE ===
    XGB --> GNN_OUT[GNN Output CSV]
    DATA -.-> MERGE
    GNN_OUT --> MERGE[Merge Pipeline]
    MERGE --> RESULT[Final Fraud Report]

    %% === WRITEBACK & BLOOM ===
    RESULT --> UPDATE[Update Neo4j Properties]
    UPDATE --> BLOOM[Neo4j Bloom Visualization]


```
---

## 📊 Neo4j Bloom Visualization Results

<table>
  <tr>
    <td width="50%">
      <h3 align="center">🔴 High Risk Fraud</h3>
      <div align="center">
        <img src="img/fraud_model.png" alt="Fraud Model" width="100%">
      </div>
      <p align="center">
        Node yang terdeteksi sebagai <b>Fraud</b> dengan probabilitas tinggi (Certainty > 0.8).
      </p>
    </td>
    <td width="50%">
      <h3 align="center">⚪ Normal Claims</h3>
      <div align="center">
        <img src="img/normal_model.png" alt="Normal Model" width="100%">
      </div>
      <p align="center">
        Klaim yang diprediksi <b>Aman/Normal</b> dengan probabilitas fraud sangat rendah.
      </p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3 align="center">🟣 Prediction Mismatch</h3>
      <div align="center">
        <img src="img/bukanFraud_fraudPrediction.png" alt="Mismatch Prediction" width="100%">
      </div>
      <p align="center">
        <b>False Positive/Negative:</b> Node dimana prediksi model berbeda dengan data aktual (Predicted != Actual).
      </p>
    </td>
    <td width="50%">
      <h3 align="center">🔘 Uncertain / Mid-Range</h3>
      <div align="center">
        <img src="img/mid_model.png" alt="Mid Model" width="100%">
      </div>
      <p align="center">
        Klaim dengan skor probabilitas di area abu-abu (0.5 - 0.8), membutuhkan investigasi manual.
      </p>
    </td>
  </tr>
</table>

---

## 🖥️ Interface Showcase

<table width="100%">
  <tr>
    <td width="50%" align="center" valign="top">
      <h3>🕸️ Neo4j Graph Exploration</h3>
      <p><b>Interactive Fraud Network Visualization</b></p>
      <img src="vid/vid1.gif" alt="Neo4j Graph Visualization" width="100%" style="border-radius: 5px;">
      <br/><br/>
      <p align="left">
        <b>🔍 Deep Forensic Investigation</b><br>
        ⚡ <b>Visual Relationship Mapping <br>
        🏷️ <b>Risk Indicator <br>
        📊 <b>Live Schema Overview <br>
        <i>Antarmuka visual yang canggih untuk mendeteksi jaringan fraud dan anomali yang kompleks melalui kueri Cypher.</i>
      </p>
    </td>
    <td width="50%" align="center" valign="top">
      <h3>📟 GRAFANA CLI Dashboard</h3>
      <p><b>Command-Line Pipeline Orchestrator</b></p>
      <img src="vid/vid2.gif" alt="GRAFANA CLI Interface" width="100%" style="border-radius: 5px;">
      <br/><br/>
      <p align="left">
        <b>🚀 Advanced Terminal Experience</b><br>
        🛠️ <b>Interactive REPL Menu. <br>
        ⚙️ <b>Dynamic Configuration. <br>
        🔌 <b>Live Connection Status. <br>
        <i>Antarmuka terminal profesional untuk mengelola seluruh siklus hidup pipeline deteksi fraud secara efisien.</i>
      </p>
    </td>
  </tr>
</table>

---

## 🧱 System Requirements

* Python ≥ 3.10
* Neo4j Desktop / Neo4j AuraDB
* CUDA (opsional, untuk training GNN)
* Pip & Virtualenv

---

## 🐍 Create Virtual Environment

```bash
git clone https://github.com/ahnafyura/GRAFANA
cd GRAFANA
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

## 📦 Install Dependencies

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

## 🏗️ Neo4j Setup

### Instalasi Neo4j Desktop

Download: [https://neo4j.com/download/](https://neo4j.com/download/)

Setelah instalasi:

1. Buat database baru
2. Username & password default:
  * neo4j
  * neo4j123
3. Jalankan database

## 🚀 Quick Start

Run interactive REPL user interface
```bash
python -m repl.repl
```

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

## 🔗 Graph Model Design

### Node Types

* **Claim**
* **Patient**
* **Diagnosis**
* **Procedure**
* **ServiceType**
* **CareClass**

## 👁️ Graph Visualization

### Neo4j Browser

Melihat 50 Claim:

```cypher
MATCH (c:Claim)-[r]-(n)
RETURN * LIMIT 50;
```

---

## 📄 **License**

MIT License
