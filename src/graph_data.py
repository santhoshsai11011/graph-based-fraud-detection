import torch
from torch_geometric.data import Data
import numpy as np

def build_graph_data(
    processed_df,
    edges_df
):

    labeled_df = (
    processed_df[
        processed_df["class"] != "unknown"
    ]
    .copy()
    .reset_index(drop=True)
    )

    labeled_df["class"] = (
        labeled_df["class"]
        .astype(int)
    )

    labeled_df["target"] = (
        labeled_df["class"]
        .map({
            1: 1,
            2: 0
        })
    )

    feature_columns = [
        col
        for col in labeled_df.columns
        if col.startswith("feature_")
    ]

    x = torch.tensor(
        labeled_df[
            feature_columns
        ].values,
        dtype=torch.float
    )

    y = torch.tensor(
        labeled_df["target"].values,
        dtype=torch.long
    )

    node_mapping = {
        tx_id: idx
        for idx, tx_id in enumerate(
            labeled_df["tx_id"]
        )
    }

    valid_edges = edges_df[
        edges_df["txId1"].isin(node_mapping)
        &
        edges_df["txId2"].isin(node_mapping)
    ]

    source_nodes = (
        valid_edges["txId1"]
        .map(node_mapping)
        .values
    )

    target_nodes = (
        valid_edges["txId2"]
        .map(node_mapping)
        .values
    )

    edge_index = torch.tensor(
    np.vstack(
        [source_nodes, target_nodes]
    ),
    dtype=torch.long
    )

    num_nodes = len(y)

    train_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool
    )

    val_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool
    )

    test_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool
    )

    train_idx = labeled_df[
        labeled_df["time_step"] <= 34
    ].index

    val_idx = labeled_df[
        (labeled_df["time_step"] >= 35)
        &
        (labeled_df["time_step"] <= 39)
    ].index

    test_idx = labeled_df[
        labeled_df["time_step"] >= 40
    ].index

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(
    "\nGraph Data Summary"
    )

    print(
    "Nodes:",
    data.num_nodes
    )

    print(
    "Edges:",
    data.num_edges
    )

    print(
    "Features:",
    data.num_node_features
    )

    print(
    "Train Nodes:",
    int(data.train_mask.sum())
    )

    print(
    "Validation Nodes:",
    int(data.val_mask.sum())
    )

    print(
    "Test Nodes:",
    int(data.test_mask.sum())
    )
    return data