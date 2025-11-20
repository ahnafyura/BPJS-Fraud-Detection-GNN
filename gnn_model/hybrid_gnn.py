import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import os

# ---------- CONFIG ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.001
EPOCHS = 300
HIDDEN_DIM = 128

# Path File (Menggunakan path relatif dari folder gnn_model)
NODES_PATH = "output/fraud_detection_final_report.csv" 
EDGES_PATH = "../data/processed/neo4j_edges.csv"

print(f"⚙️  Running on DEVICE: {DEVICE}")

# ---------- 1. LOAD DATA ----------
print("📂 Loading Data...")

# Cek apakah file ada
if not os.path.exists(NODES_PATH):
    # Coba path alternatif jika dijalankan dari root
    NODES_PATH = "../gnn_model/output/gnn_hybrid_predicted_nodes.csv"
    if not os.path.exists(NODES_PATH):
        raise FileNotFoundError(f"❌ Tidak bisa menemukan file nodes di: {NODES_PATH}")

nodes_df = pd.read_csv(NODES_PATH)
edges_df = pd.read_csv(EDGES_PATH)

# Pastikan mapped_id urut
nodes_df = nodes_df.sort_values('mapped_id').reset_index(drop=True)
N = len(nodes_df)
print(f"   Nodes: {N}, Edges: {len(edges_df)}")

# ---------- 2. PREPARE FEATURES (X) ----------
exclude_cols = [
    'node_id', 'mapped_id', 'is_fraud', 'train_mask', 'test_mask', 
    'fraud_score', 'final_risk_score', 'ai_explanation', 'fraud_reason',
    'new_gnn_score' # Hindari membaca hasil prediksi diri sendiri jika ada
]

feature_cols = [c for c in nodes_df.columns if c not in exclude_cols and np.issubdtype(nodes_df[c].dtype, np.number)]
print(f"📊 Using {len(feature_cols)} Features.")

X_raw = nodes_df[feature_cols].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
x = torch.tensor(X_scaled, dtype=torch.float, device=DEVICE)

# ---------- 3. PREPARE LABELS (y) ----------
# Pastikan label ada isinya (handle null dengan 0)
y_raw = nodes_df['is_fraud'].fillna(0).values
y = torch.tensor(y_raw, dtype=torch.float, device=DEVICE)

# Cek jumlah fraud
fraud_count = y.sum().item()
print(f"   Total Fraud di Data: {int(fraud_count)} dari {N} nodes.")

if fraud_count < 2:
    print("⚠️ WARNING: Jumlah Fraud terlalu sedikit (<2). AUC tidak akan valid.")

# ---------- 4. PREPARE EDGE INDEX ----------
# Mapping ID sederhana
if 'source_mapped' in edges_df.columns:
    src = edges_df['source_mapped']
    dst = edges_df['target_mapped']
else:
    # Fallback mapping
    nid_to_mid = dict(zip(nodes_df['node_id'], nodes_df['mapped_id']))
    src = edges_df['source'].map(nid_to_mid).fillna(-1).astype(int)
    dst = edges_df['target'].map(nid_to_mid).fillna(-1).astype(int)

mask_edge = (src >= 0) & (dst >= 0)
edge_index = torch.tensor([src[mask_edge].values, dst[mask_edge].values], dtype=torch.long, device=DEVICE)

# ---------- 5. FORCE STRATIFIED SPLIT (Perbaikan Utama) ----------
# Kita buat mask baru secara paksa agar Fraud terbagi rata
print("🔄 Performing Stratified Split (70:30)...")
try:
    train_idx, test_idx = train_test_split(
        np.arange(N), 
        test_size=0.3, 
        stratify=y_raw,
        random_state=42
    )
except ValueError:
    print("⚠️ Gagal Stratify (mungkin fraud cuma 1). Fallback ke Random Split.")
    train_idx, test_idx = train_test_split(np.arange(N), test_size=0.3, random_state=42)

train_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
test_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
train_mask[train_idx] = True
test_mask[test_idx] = True

print(f"   Train Fraud: {y[train_mask].sum().item()} | Test Fraud: {y[test_mask].sum().item()}")

# ---------- 6. DEFINE MODEL ----------
class HybridGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.fc(x)
        return x.squeeze()

model = HybridGNN(in_channels=x.shape[1], hidden_channels=HIDDEN_DIM, out_channels=1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# Calculate Imbalance Weight
num_pos = y[train_mask].sum().item()
num_neg = (~y[train_mask].bool()).sum().item()
pos_weight = torch.tensor([num_neg / max(1, num_pos)], device=DEVICE)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ---------- 7. TRAINING LOOP ----------
print("\n🚀 Start Training...")
best_auc = 0
patience = 30
patience_counter = 0
model_saved = False

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            pred_logits = model(x, edge_index)
            pred_probs = torch.sigmoid(pred_logits)
            y_true_test = y[test_mask].cpu().numpy()
            y_pred_test = pred_probs[test_mask].cpu().numpy()
            
            try:
                # Safety check untuk AUC
                if len(np.unique(y_true_test)) > 1:
                    auc = roc_auc_score(y_true_test, y_pred_test)
                    pr = average_precision_score(y_true_test, y_pred_test)
                else:
                    auc = 0.5 # Default random guess
                    pr = 0.0
            except:
                auc = 0.0
                pr = 0.0

            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Test AUC: {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                patience_counter = 0
                torch.save(model.state_dict(), "best_hybrid_gnn.pth")
                model_saved = True
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print("⏹️  Early stopping triggered.")
                break

# ---------- 8. FINAL OUTPUT ----------
print("\n✅ Training Finished.")

# Safety Load: Hanya load jika file benar-benar ada
if model_saved and os.path.exists("best_hybrid_gnn.pth"):
    print("💾 Loading best model saved during training...")
    model.load_state_dict(torch.load("best_hybrid_gnn.pth"))
else:
    print("⚠️ Warning: Tidak ada model terbaik yang disimpan (AUC stagnan). Menggunakan model terakhir.")

model.eval()
with torch.no_grad():
    final_logits = model(x, edge_index)
    final_probs = torch.sigmoid(final_logits).cpu().numpy()

nodes_df['new_gnn_score'] = final_probs
OUTPUT_FILE = "output/gnn_retrained_output.csv"

# Pastikan folder output ada
os.makedirs("output", exist_ok=True)
nodes_df.to_csv(OUTPUT_FILE, index=False)
print(f"📂 Hasil tersimpan di: {os.path.abspath(OUTPUT_FILE)}")