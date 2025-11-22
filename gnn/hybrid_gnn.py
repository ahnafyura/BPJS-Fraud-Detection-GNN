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

def run():
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
                raise FileNotFoundError(f"Could not find file at: {NODES_PATH}")

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
    included_cols = [
        'closeness', 'degree', 
        'community_density','betweenness',
        
        'tarif_seharusnya','tarif_diklaim'
    ]

    def prepare_features(included_cols):
        feature_cols = [c for c in nodes_df.columns if c in included_cols and np.issubdtype(nodes_df[c].dtype, np.number)] # type: ignore
        print(f"Using {len(feature_cols)} Features.")
        print(f"Features: {feature_cols}")

        X_raw = nodes_df[feature_cols].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        return torch.tensor(X_scaled, dtype=torch.float, device=DEVICE)

    x = prepare_features(included_cols=included_cols)

    # ---------- 3. PREPARE LABELS (y) ----------
    def prepare_labels():
        # 1️⃣ Labels
        y_raw = nodes_df['is_fraud'].fillna(0).values
        y = torch.tensor(y_raw, dtype=torch.float, device=DEVICE)

        fraud_count = y.sum().item()
        print(f"Total Fraud in Data: {int(fraud_count)} from {len(nodes_df)} nodes.")

        # 2️⃣ Map node_id to 0..N-1
        nid_to_idx = {nid: i for i, nid in enumerate(nodes_df['node_id'])}

        # Apply mapping to edges
        src = edges_df['source'].map(nid_to_idx) # type: ignore
        dst = edges_df['target'].map(nid_to_idx) # type: ignore

        # Remove any edges with missing nodes
        mask = src.notna() & dst.notna()
        src = src[mask].astype(int)
        dst = dst[mask].astype(int)

        # 3️⃣ Build edge_index
        edge_index = torch.tensor([src.values, dst.values], dtype=torch.long, device=DEVICE) # type: ignore

        return y_raw, y, edge_index



    y_raw, y, edge_index = prepare_labels()
        

    # ---------- 5. FORCE STRATIFIED SPLIT (Perbaikan Utama) ----------

    def force_stratified_split():
        # Kita buat mask baru secara paksa agar Fraud terbagi rata
        print("Performing Stratified Split...")
        try:
            train_idx, test_idx = train_test_split(
                np.arange(N), 
                test_size=env.test_size, 
                stratify=y_raw,
                random_state=42
            )
        except ValueError:
            print("Stratify failure. Fallback to Random Split.")
            train_idx, test_idx = train_test_split(np.arange(N), test_size=env.test_size, random_state=42)
        
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
    best_auc = 0
    patience = 30
    patience_counter = 0

    if (not env.skip_gnn_training):
        print("\nStart Training...")

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

                    print(f"Test AUC: {auc:.4f}")


                    print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Test AUC: {auc:.4f}")
                    
                    if auc > best_auc:
                        best_auc = auc
                        patience_counter = 0
                        torch.save(model.state_dict(), env.BEST_GNN_HYBRID_PATH)
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break
        print("\nTraining Finished.")
        print(f"Best AUC: {best_auc}")

    # ---------- 8. FINAL OUTPUT ----------

    # Safety Load: Hanya load jika file benar-benar ada
    if os.path.exists(env.BEST_GNN_HYBRID_PATH):
        print("Loading saved model.")
        model.load_state_dict(torch.load(env.BEST_GNN_HYBRID_PATH))
    else:
        print("Warning: No best model is saved. Using last model...")

    model.eval()
    with torch.no_grad():
        final_logits = model(x, edge_index)
        final_probs = torch.sigmoid(final_logits).cpu().numpy()

    nodes_df['fraud_certainty'] = final_probs

    # Pastikan folder output ada
    os.makedirs("output", exist_ok=True)
    nodes_df.to_csv(env.RETRAINED_OUTPUT_FILE , index=False)

    claim_nodes = nodes_df[nodes_df['labels'].str.contains('Claim', na=False)]

    # Boolean column (True/False)
    claim_nodes = nodes_df[nodes_df['labels'].str.contains("Claim")].copy()
    claim_nodes['predicted_fraud'] = np.round(claim_nodes['fraud_certainty']).astype(int)
    claim_nodes[['fraud_certainty', 'predicted_fraud']].head() # type: ignore

    accuracy = (claim_nodes['predicted_fraud'] == claim_nodes['is_fraud']).mean()
    print(f"Accuracy: {accuracy:.4f}")

    claims_cols = [
        'node_id', 'tarif_seharusnya', 
        'fraud_type', 'tarif_diklaim', 'catatan', 'lama_rawat', 
        'is_fraud', 'status_klaim', 'id_klaim', 'fraud_certainty', 'predicted_fraud']
    claim_nodes.to_csv(env.RESULTS_CLAIMS_FILE, columns=claims_cols, index=False)
    print(f"Output saved at: {os.path.abspath(env.RETRAINED_OUTPUT_FILE)}")

if __name__ == "__main__":
    run()
