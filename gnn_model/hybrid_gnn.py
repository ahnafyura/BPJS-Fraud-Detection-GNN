# hybrid_gnn.py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import community as community_louvain  # python-louvain, for recomputing communities in Python if needed

# ---------- CONFIG ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA_COMM = 1.0     # weight for community-consistency loss; tune this
EMB_DIM = 64
EPOCHS = 1000
LR = 1e-3

# ---------- LOAD CSVs (paths you already have) ----------
# all_nodes should contain: node_id, mapped_id, label (Claim/Patient..), is_fraud, community (optional), community_size
all_nodes = pd.read_csv("neo4j_query_table_data_2025-11-19.csv")  # or 'all_nodes.csv'
edges = pd.read_csv("graf_table.csv")  # should contain source_mapped, target_mapped

# NOTE: ensure mapped ids exist (mapped to contiguous 0..N-1)
# If not yet mapped, create mapping:
if "mapped_id" not in all_nodes.columns:
    mapping = {nid: i for i, nid in enumerate(all_nodes['node_id'].tolist())}
    all_nodes['mapped_id'] = all_nodes['node_id'].map(mapping)
    edges['source_mapped'] = edges['source'].map(mapping)
    edges['target_mapped'] = edges['target'].map(mapping)

N = len(all_nodes)
print("Nodes:", N, "Edges:", len(edges))

# ---------- FEATURE PREP ----------
# choose numeric features; do NOT include raw community id as numeric
num_cols = ["tarif_diklaim", "tarif_seharusnya", "lama_rawat", "community_size"]
for col in num_cols:
    if col not in all_nodes.columns:
        all_nodes[col] = 0
X_num = all_nodes[num_cols].fillna(0).values
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)

# optionally add node degree / pagerank if present:
if "degree" in all_nodes.columns:
    deg = all_nodes["degree"].fillna(0).values.reshape(-1,1)
    X = np.hstack([X_num, deg])
else:
    X = X_num

x = torch.tensor(X, dtype=torch.float, device=DEVICE)

# ---------- LABELS ----------
# We'll predict is_fraud for Claim nodes; keep labels for all nodes (non-claim can be 0)
y = torch.tensor(all_nodes["is_fraud"].fillna(0).astype(int).values, dtype=torch.long, device=DEVICE)

# ---------- EDGE index ----------
edge_index = torch.tensor(edges[["source_mapped","target_mapped"]].values.T, dtype=torch.long).to(DEVICE)
# make undirected
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

# ---------- TRAIN / TEST MASKS ----------
# assume you have boolean masks columns, or lists of mapped ids for train/test
# if you don't, create masks from train_nodes/test_nodes file created earlier
try:
    train_nodes = pd.read_csv("train_nodes.csv")
    test_nodes = pd.read_csv("test_nodes.csv")
    train_idx = train_nodes["mapped_id"].values
    test_idx = test_nodes["mapped_id"].values
except Exception:
    # fallback: random stratified split on Claim nodes
    from sklearn.model_selection import train_test_split
    claim_rows = all_nodes.index[all_nodes['label']=="Claim"].tolist()
    claim_labels = all_nodes.loc[claim_rows, "is_fraud"].fillna(0).astype(int).values
    train_claim_idx, test_claim_idx = train_test_split(claim_rows, test_size=0.3, stratify=claim_labels, random_state=42)
    train_idx = np.array(train_claim_idx)
    test_idx = np.array(test_claim_idx)

train_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
test_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
train_mask[train_idx] = True
test_mask[test_idx] = True

# ---------- COMMUNITY PSEUDO-LABELS (IMPORTANT: compute on training subgraph) ----------
# If you ran Louvain in Neo4j only on full graph, recompute on training subgraph to avoid leakage:
# Recompute communities for the training subgraph using python-louvain on adjacency restricted to train nodes

def recompute_louvain_on_train(edges_df, train_idx):
    # build adjacency on train nodes
    train_set = set(train_idx.tolist())
    # adjacency list
    import networkx as nx
    G = nx.Graph()
    # add only train nodes
    G.add_nodes_from(train_idx.tolist())
    # add edges where both endpoints in train_set
    for _, row in edges_df.iterrows():
        a = int(row['source_mapped']); b = int(row['target_mapped'])
        if a in train_set and b in train_set:
            G.add_edge(a,b)
    if G.number_of_edges() == 0:
        return {}
    partition = community_louvain.best_partition(G)  # dict: node -> community id
    return partition

partition = recompute_louvain_on_train(edges, train_idx)
# create community array aligned to mapped indices (only for train nodes; others = -1)
community_array = -1 * np.ones(N, dtype=int)
if partition:
    for node_mapped, comm in partition.items():
        community_array[int(node_mapped)] = int(comm)

# store in tensor
community_tensor = torch.tensor(community_array, dtype=torch.long, device=DEVICE)

# ---------- MODEL: GraphSAGE backbone + classifier head ----------------
class HybridGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden=EMB_DIM, out_classes=2):
        super().__init__()
        self.sage1 = SAGEConv(in_channels, hidden)
        self.sage2 = SAGEConv(hidden, hidden//2)
        self.proj = torch.nn.Linear(hidden//2, hidden//2)  # optional projection head
        self.clf = torch.nn.Linear(hidden//2, out_classes)

    def forward(self, x, edge_index):
        h = F.relu(self.sage1(x, edge_index))
        h = F.relu(self.sage2(h, edge_index))
        z = self.proj(h)  # embedding used for both classifier and community loss
        logits = self.clf(z)
        return logits, z

model = HybridGNN(x.shape[1]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
ce_loss = torch.nn.CrossEntropyLoss()

# ---------- COMMUNITY-CONSISTENCY LOSS helper ----------
def community_consistency_loss(z, community_array, train_mask_bool):
    # compute centroid for each community among train nodes
    # community_array: np array with -1 for unlabeled nodes
    device = z.device
    loss = torch.tensor(0.0, device=device)
    unique_comms = np.unique(community_array[community_array >= 0])
    if len(unique_comms) == 0:
        return loss
    # for each community compute centroid and add squared distances
    count = 0
    for c in unique_comms:
        idxs = np.where(community_array == c)[0]
        # ensure they are in train set
        idxs = [i for i in idxs if train_mask_bool[i]]
        if len(idxs) <= 1:
            continue
        idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
        zc = z[idxs_t]
        centroid = zc.mean(dim=0, keepdim=True)
        loss += ((zc - centroid) ** 2).sum()
        count += len(idxs)
    if count == 0:
        return torch.tensor(0.0, device=device)
    return loss / count

# ---------- TRAIN LOOP ----------
print("Start training hybrid GNN (CE + community consistency)...")
train_mask_bool = train_mask.cpu().numpy().astype(bool)

for epoch in range(EPOCHS):
    model.train()
    opt.zero_grad()
    logits, z = model(x, edge_index)
    loss_ce = ce_loss(logits[train_mask], y[train_mask])

    # community loss computed only for train nodes that have community labels
    comm_loss = community_consistency_loss(z, community_array, train_mask_bool)
    loss = loss_ce + LAMBDA_COMM * comm_loss
    loss.backward()
    opt.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            logits_eval, _ = model(x, edge_index)
            preds = logits_eval.argmax(dim=1)
            y_true = y[test_mask].cpu().numpy()
            y_pred = preds[test_mask].cpu().numpy()
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"Epoch {epoch:03d} loss_ce {loss_ce.item():.4f} comm_loss {comm_loss.item():.4f} val_acc {acc:.4f} val_f1 {f1:.4f}")

# ---------- FINAL EVALUATION ----------
model.eval()
with torch.no_grad():
    logits_final, z_final = model(x, edge_index)
    probs = F.softmax(logits_final, dim=1)[:,1].cpu().numpy()
    preds = logits_final.argmax(dim=1).cpu().numpy()

y_true = y[test_mask].cpu().numpy()
y_pred = preds[test_mask]
auc = roc_auc_score(y_true, logits_final[test_mask][:,1].cpu().numpy())
print("Final AUC:", auc)

# save predicted scores back to CSV
all_nodes["mapped_id"] = all_nodes["mapped_id"].astype(int)
all_nodes["fraud_score"] = probs
all_nodes.to_csv("gnn_hybrid_predicted_nodes.csv", index=False)

print("Saved gnn_hybrid_predicted_nodes.csv")