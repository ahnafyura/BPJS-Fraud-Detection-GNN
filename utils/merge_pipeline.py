#!/usr/bin/env python3
"""
utils/merge_pipeline.py

ROBUST MERGE PIPELINE
---------------------
Fungsi: Menggabungkan hasil output GNN (yang mungkin mengandung node Patient/Diagnosis)
kembali ke Data Klaim Asli (Raw CSV) berdasarkan ID Klaim.

Fitur Utama:
1. Auto-filter: Hanya mengambil baris dengan label "Claim" dari output GNN.
2. ID Sanitization: Membersihkan whitespace pada ID agar matching akurat.
3. Duplicate Handling: Jika ID ganda di GNN, ambil certainty tertinggi.
4. Safe Join: Menggunakan Left Join agar data asli tidak berkurang.

Usage:
python utils/merge_pipeline.py \
  --original data/raw/claims_synthetic_840.csv \
  --gnn output/gnn_retrained_output.csv \
  --out output/FINAL_REPORT.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime

def safe_load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    try:
        df = pd.read_csv(path)
        # Bersihkan nama kolom dari spasi berlebih
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        raise ValueError(f"Gagal membaca CSV {path}: {e}")

def normalize_id_column(df: pd.DataFrame, col_name: str):
    """Membersihkan kolom ID menjadi string bersih tanpa spasi."""
    df[col_name] = df[col_name].astype(str).str.strip()
    return df

def detect_id_col(df: pd.DataFrame, candidates=("ID Klaim", "id_klaim", "claim_id", "id")) -> str:
    """Mencoba mendeteksi nama kolom ID secara otomatis."""
    cols = list(df.columns)
    # Cek kandidat prioritas
    for c in candidates:
        if c in cols:
            return c
    # Cek pattern string
    for c in cols:
        if "klaim" in c.lower() or "claim" in c.lower():
            return c
    raise ValueError(f"Tidak dapat mendeteksi kolom ID Klaim. Kolom tersedia: {cols}")

def unique_outpath(path: str) -> str:
    """Membuat nama file unik agar tidak menimpa file lama."""
    base, ext = os.path.splitext(path)
    if not os.path.exists(path):
        return path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}{ext}"

def merge_and_save(original_path: str,
                   gnn_path: str,
                   out_path: str = None,
                   id_original: str = None,
                   id_gnn: str = None,
                   threshold: float = 0.5,
                   save_xlsx: bool = False,
                   verbose: bool = True):

    # 1. LOAD DATA
    if verbose: print(f"📂 Loading files...")
    df_raw = safe_load_csv(original_path)
    df_gnn = safe_load_csv(gnn_path)

    # 2. DETEKSI ID KOLOM
    if id_original is None:
        id_original = detect_id_col(df_raw)
    if id_gnn is None:
        id_gnn = detect_id_col(df_gnn)

    if verbose:
        print(f"   🔹 Raw Rows: {len(df_raw):,} | ID Col: '{id_original}'")
        print(f"   🔹 GNN Rows: {len(df_gnn):,} | ID Col: '{id_gnn}'")

    # 3. FILTERING GNN (CRITICAL STEP!)
    # Hanya ambil baris yang merupakan 'Claim'. Membuang Patient, Diagnosis, dll.
    # Ini mencegah ID kosong (NaN) dari node lain merusak hasil merge.
    if 'labels' in df_gnn.columns:
        initial_len = len(df_gnn)
        # Filter baris yang kolom 'labels'-nya mengandung kata 'Claim'
        df_gnn_claims = df_gnn[df_gnn['labels'].astype(str).str.contains('Claim', case=False, na=False)].copy()
        filtered_len = len(df_gnn_claims)
        if verbose:
            print(f"🧹 Filtering Non-Claim Nodes: {initial_len} -> {filtered_len} rows (Removed {initial_len - filtered_len} junk nodes)")
    else:
        print("⚠ WARNING: Kolom 'labels' tidak ditemukan di output GNN. Asumsi semua baris adalah klaim.")
        df_gnn_claims = df_gnn.copy()

    # 4. DETEKSI KOLOM CERTAINTY
    possible_cert_names = ["fraud_certainty", "fraud_score", "probability", "fraud_prob"]
    cert_col = None
    for c in possible_cert_names:
        if c in df_gnn_claims.columns:
            cert_col = c
            break
    
    if cert_col is None:
        # Fallback: cari kolom float selain ID
        numeric_cols = df_gnn_claims.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            cert_col = numeric_cols[-1] # Ambil kolom numeric terakhir (biasanya output layer)
    
    if cert_col is None:
        raise ValueError("❌ Tidak dapat menemukan kolom Fraud Certainty di file GNN.")
    
    if verbose: print(f"   🔹 Using Certainty Column: '{cert_col}'")

    # 5. STANDARISASI & PEMBERSIHAN ID
    df_raw = normalize_id_column(df_raw, id_original)
    df_gnn_claims = normalize_id_column(df_gnn_claims, id_gnn)

    # 6. HANDLING DUPLICATES DI GNN
    # Jika satu klaim punya 2 prediksi (jarang, tapi bisa terjadi), ambil yang skornya tertinggi
    if df_gnn_claims.duplicated(subset=[id_gnn]).any():
        if verbose: print("   ⚠ Ditemukan duplikasi ID di GNN. Mengambil nilai certainty tertinggi...")
        df_gnn_claims = df_gnn_claims.sort_values(cert_col, ascending=False).drop_duplicates(subset=[id_gnn])

    # 7. PERSIAPAN DATA UNTUK MERGE
    # Kita hanya ambil kolom ID dan Certainty dari GNN
    # (Bisa tambah kolom lain di 'cols_to_keep' jika perlu, misal 'community_id')
    cols_to_keep = [id_gnn, cert_col]
    df_gnn_subset = df_gnn_claims[cols_to_keep].copy()
    
    # Rename agar tidak bentrok dan jelas
    df_gnn_subset = df_gnn_subset.rename(columns={
        id_gnn: "join_key_id",
        cert_col: "fraud_certainty"
    })

    # 8. EKSEKUSI MERGE (LEFT JOIN)
    if verbose: print("🔗 Merging data (Left Join)...")
    
    df_final = pd.merge(
        df_raw,
        df_gnn_subset,
        left_on=id_original,
        right_on="join_key_id",
        how="left"
    )

    # 9. POST-PROCESSING
    # Isi NaN dengan 0.0 (untuk klaim yang tidak ada di graf/GNN)
    missing_count = df_final['fraud_certainty'].isna().sum()
    if missing_count > 0 and verbose:
        print(f"   ℹ {missing_count} klaim tidak memiliki prediksi dari GNN (Default set ke 0.0)")
    
    df_final['fraud_certainty'] = df_final['fraud_certainty'].fillna(0.0)
    
    # Buat kolom prediksi binary (0/1)
    df_final['predicted_fraud'] = (df_final['fraud_certainty'] > threshold).astype(int)

    # Hapus kolom bantuan join_key
    if "join_key_id" in df_final.columns:
        df_final.drop(columns=["join_key_id"], inplace=True)

    # 10. SIMPAN HASIL
    os.makedirs("output", exist_ok=True)
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"output/final_report_{ts}.csv"
    
    out_path = unique_outpath(out_path)
    
    df_final.to_csv(out_path, index=False)
    print(f"✅ SUKSES! File tersimpan di: {out_path}")

    if save_xlsx:
        xlsx_path = out_path.replace(".csv", ".xlsx")
        df_final.to_excel(xlsx_path, index=False)
        print(f"✅ Excel version saved: {xlsx_path}")

    # 11. PREVIEW STATISTIK
    fraud_count = df_final['predicted_fraud'].sum()
    total_count = len(df_final)
    print("-" * 40)
    print(f"📊 SUMMARY:")
    print(f"   Total Klaim     : {total_count}")
    print(f"   Prediksi Fraud  : {fraud_count} ({fraud_count/total_count:.1%})")
    print(f"   Avg Certainty   : {df_final['fraud_certainty'].mean():.4f}")
    print("-" * 40)

def build_cli():
    p = argparse.ArgumentParser(description="Robust Merge Pipeline for GNN Results")
    p.add_argument("--original", required=True, help="Path file CSV data asli (Data 2)")
    p.add_argument("--gnn", required=True, help="Path file CSV output GNN (Data 1)")
    p.add_argument("--out", default=None, help="Nama file output (opsional)")
    p.add_argument("--id-original", default=None, help="Nama kolom ID di file asli")
    p.add_argument("--id-gnn", default=None, help="Nama kolom ID di file GNN")
    p.add_argument("--threshold", type=float, default=0.5, help="Batas threshold fraud (default 0.5)")
    p.add_argument("--xlsx", action="store_true", help="Simpan juga versi Excel (.xlsx)")
    return p

if __name__ == "__main__":
    args = build_cli().parse_args()
    merge_and_save(
        original_path=args.original,
        gnn_path=args.gnn,
        out_path=args.out,
        id_original=args.id_original,
        id_gnn=args.id_gnn,
        threshold=args.threshold,
        save_xlsx=args.xlsx
    )