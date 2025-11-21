import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import os
from config import env

# ---------- CONFIG ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.001
EPOCHS = 300
HIDDEN_DIM = 128

# Path File (Menggunakan path relatif dari folder gnn_model)
NODES_PATH = env.NODES_CSV
EDGES_PATH = env.EDGES_CSV

print(f"Running on DEVICE: {DEVICE}")

# ---------- 1. LOAD DATA ----------

def load_data(NODES_PATH):
    print("Loading Data...")

    # Cek apakah file ada
    if not os.path.exists(NODES_PATH):
        if not os.path.exists(NODES_PATH):
            raise FileNotFoundError(f"Tidak bisa menemukan file nodes di: {NODES_PATH}")

    nodes_df = pd.read_csv(NODES_PATH)
    edges_df = pd.read_csv(EDGES_PATH)

    print(nodes_df['is_fraud'].value_counts())

    # Pastikan mapped_id urut
    # nodes_df = nodes_df.sort_values('mapped_id').reset_index(drop=True)
    N = len(nodes_df)
    print(f"   Nodes: {N}, Edges: {len(edges_df)}")

    return nodes_df, edges_df, N

nodes_df, edges_df, N = load_data(NODES_PATH=NODES_PATH)

# ---------- 2. PREPARE FEATURES (X) ----------
exclude_cols = [
    'community_id', 'node_id', 'mapped_id', 'is_fraud', 'train_mask', 'test_mask', 
    'fraud_score', 'final_risk_score', 'ai_explanation', 'fraud_reason',
    'new_gnn_score',  # Hindari membaca hasil prediksi diri sendiri jika ada
]

def prepare_features(exclude_cols):
    feature_cols = [c for c in nodes_df.columns if c not in exclude_cols and np.issubdtype(nodes_df[c].dtype, np.number)] # type: ignore
    print(f"Using {len(feature_cols)} Features.")
    print(f"Features: {feature_cols}")

    X_raw = nodes_df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return torch.tensor(X_scaled, dtype=torch.float, device=DEVICE)

x = prepare_features(exclude_cols=exclude_cols)

# ---------- 3. PREPARE LABELS (y) ----------
def prepare_labels():
    y_raw = nodes_df['is_fraud'].fillna(0).values
    y = torch.tensor(y_raw, dtype=torch.float, device=DEVICE)

    fraud_count = y.sum().item()
    print(f"Total Fraud di Data: {int(fraud_count)} dari {len(nodes_df)} nodes.")

    src = edges_df['source']
    dst = edges_df['target']
    edge_index = torch.tensor([src.values, dst.values], dtype=torch.long, device=DEVICE)

    return y_raw, y, edge_index


y_raw, y, edge_index = prepare_labels()
    

# ---------- 5. FORCE STRATIFIED SPLIT (Perbaikan Utama) ----------

def force_stratified_split():
    # Kita buat mask baru secara paksa agar Fraud terbagi rata
    print("Performing Stratified Split...")
    try:
        train_idx, test_idx = train_test_split(
            np.arange(N), 
            test_size=env.TEST_SIZE, 
            stratify=y_raw,
            random_state=42
        )
    except ValueError:
        print("Gagal Stratify (mungkin fraud cuma 1). Fallback ke Random Split.")
        train_idx, test_idx = train_test_split(np.arange(N), test_size=env.TEST_SIZE, random_state=42)
    
    train_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    test_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    print(f"   Train Fraud: {y[train_mask].sum().item()} | Test Fraud: {y[test_mask].sum().item()}")

    return train_mask, test_mask

train_mask, test_mask = force_stratified_split()

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

def create_model():
    model = HybridGNN(in_channels=x.shape[1], hidden_channels=HIDDEN_DIM, out_channels=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # Calculate Imbalance Weight
    num_pos = y[train_mask].sum().item()
    num_neg = (~y[train_mask].bool()).sum().item()
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], device=DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    return model, optimizer, criterion

model, optimizer, criterion = create_model()

# ---------- 7. TRAINING LOOP ----------
print("\nStart Training...")
best_auc = 0
patience = 100
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
                torch.save(model.state_dict(), env.BEST_GNN_HYBRID_PATH)
                model_saved = True
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

# ---------- 8. FINAL OUTPUT ----------
print("\nTraining Finished.")

# Safety Load: Hanya load jika file benar-benar ada
if model_saved and os.path.exists(env.BEST_GNN_HYBRID_PATH):
    print("Loading best model saved during training...")
    model.load_state_dict(torch.load(env.BEST_GNN_HYBRID_PATH))
else:
    print("Warning: Tidak ada model terbaik yang disimpan (AUC stagnan). Menggunakan model terakhir.")

model.eval()
with torch.no_grad():
    final_logits = model(x, edge_index)
    final_probs = torch.sigmoid(final_logits).cpu().numpy()

nodes_df['new_gnn_score'] = final_probs

# Pastikan folder output ada
os.makedirs("output", exist_ok=True)
nodes_df.to_csv(env.OUTPUT_FILE, index=False)
print(f"Hasil tersimpan di: {os.path.abspath(env.OUTPUT_FILE)}")