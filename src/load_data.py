import pandas as pd
from config import FEATURES_FILE, CLASSES_FILE, EDGES_FILE


def load_features():
    return pd.read_csv(FEATURES_FILE, header=None)


def load_classes():
    return pd.read_csv(CLASSES_FILE)


def load_edges():
    return pd.read_csv(EDGES_FILE)


def load_all_data():
    features = load_features()
    classes = load_classes()
    edges = load_edges()

    return features, classes, edges