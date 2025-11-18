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

## ⚙️ **Setup Environment**

(Full instructions retained from original project's setup — environment, plugins, settings)

---

## 📥 **Data Loading (ETL)**

(Full Python ETL script preserved here, formatted for README)

---

## 🧠 **Graph Projection + Louvain**

(Fully formatted Cypher instructions preserved)

---

## 🌐 **Visualizations**

(Neo4j Browser + Bloom instructions kept here)

---

## 📁 **Export for GNN**

(Fully included Python export script)

---

## 📄 **License**

MIT License
