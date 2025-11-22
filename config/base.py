# ======================
# NEO4J CONNECTION CONFIG
# ======================

# default credentials
url = "bolt://localhost:7687"
uname = "neo4j"
pw = "neo4j123"

# ======================
# PATH CONFIG
# ======================
RAW_INPUT_DATA = "./data/raw/claims_synthetic_840.csv"

DATA_PROCESSED_OUT_DIR = "./data/processed/"

NODES_CSV = "./data/processed/nodes.csv"
EDGES_CSV = "./data/processed/edges.csv"

BEST_GNN_HYBRID_PATH = "./output/best_hybrid_gnn.pth"

RETRAINED_OUTPUT_FILE = "./output/gnn_retrained_output.csv"
RESULTS_CLAIMS_FILE = "./output/claims.csv"

# ======================
# TRAINING CONFIG
# ======================

TEST_SIZE = 0.3
SKIP_GNN_TRAINING = False
