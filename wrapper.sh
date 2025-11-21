#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "=== RUN PIPELINE from $ROOT ==="

# 1) Aktifkan virtualenv bila ada
if [ -f "$ROOT/gnn_env/bin/activate" ]; then
  echo "[venv] activating gnn_env..."
  # shellcheck source=/dev/null
  source "$ROOT/gnn_env/bin/activate"
else
  echo "[venv] gnn_env not found -> using system python (warning)"
fi

# 2) ETL load -> louvain -> export -> gnn
echo "[1/5] ETL: load data -> Neo4j"
python -m etl.load

echo "[2/5] Louvain (community detection)"
python -m louvain.louvain

echo "[3/5] ETL: export nodes/edges from Neo4j"
python -m etl.export

echo "[4/5] GNN training / inference"
python -m gnn.hybrid_gnn

# 3) Merge step (utils)
echo "[5/5] Merge GNN output back to original claims (utils/merge_pipeline.py)"

# ensure utils is importable as package (safe no-op if exists)
if [ ! -f "$ROOT/utils/__init__.py" ]; then
  touch "$ROOT/utils/__init__.py"
fi

# Default paths (ubah jika perlu)
ORIG="$ROOT/data/raw/claims_synthetic_840.csv"
GNN_OUT="$ROOT/output/gnn_retrained_output.csv"
OUT_PATH="$ROOT/output/merged_$(date +%Y%m%d_%H%M%S).csv"

# If your merge script must be run as module (python -m), you can:
# python -m utils.merge_pipeline --original "$ORIG" --gnn "$GNN_OUT" --out "$OUT_PATH" --xlsx
# We'll call the script file directly for maximum compatibility:
python "$ROOT/utils/merge_pipeline.py" --original "$ORIG" --gnn "$GNN_OUT" --out "$OUT_PATH" --xlsx

echo "=== Pipeline complete ==="
echo "Merged output: $OUT_PATH"

# 4) Deactivate venv (if activated)
if [ -n "${VIRTUAL_ENV-}" ]; then
  deactivate || true
fi
