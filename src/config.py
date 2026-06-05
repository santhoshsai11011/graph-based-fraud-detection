from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"

FEATURES_FILE = DATA_DIR / "elliptic_txs_features.csv"
CLASSES_FILE = DATA_DIR / "elliptic_txs_classes.csv"
EDGES_FILE = DATA_DIR / "elliptic_txs_edgelist.csv"