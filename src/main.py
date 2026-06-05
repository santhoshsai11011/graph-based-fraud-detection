from load_data import load_all_data
from preprocess import preprocess_data, validate_data

from graph_features import (
    create_graph_feature_dataset
)

from pathlib import Path

from graph_builder import build_transaction_graph

from graph_analysis import (
    basic_graph_stats,
    connected_component_stats,
    degree_statistics,
    clustering_statistics
)

from prepare_ml_data import (
    prepare_dataset
)

from logistic_model import (
    train_logistic_temporal,
    save_model
)

from xgboost_model import (
    train_xgboost_temporal,
    save_xgboost
)

from feature_importance import (
    plot_feature_importance
)

from temporal_split import (temporal_split)

from graph_data import (
    build_graph_data
)

from graphsage_model import (
    GraphSAGE
)

from train_graphsage import (
    train_graphsage
)

def main():
    print("\nLoading Dataset...\n")

    features, classes, edges = load_all_data()

    processed_df = preprocess_data(
        features,
        classes
    )

    print("Dataset Loaded Successfully\n")
    print(processed_df.columns[:15])
    print(f"Features Shape: {features.shape}")
    print(f"Classes Shape: {classes.shape}")
    print(f"Edges Shape: {edges.shape}")

    print(f"\nProcessed Dataset Shape: {processed_df.shape}")

    validate_data(processed_df)

    print("\nBuilding Graph...")
    G = build_transaction_graph(edges)

    graph_stats = basic_graph_stats(G)

    print("\nGraph Statistics")
    print("-" * 40)

    for key, value in graph_stats.items():
        print(f"{key}: {value}")

    component_stats = connected_component_stats(G)

    print("\nConnected Components")
    print("-" * 40)
    for key, value in component_stats.items():
        print(f"{key}: {value}")

    degree_stats = degree_statistics(G)
    print("\nDegree Statistics")
    print("-" * 40)
    for key, value in degree_stats.items():
        print(f"{key}: {value}")

    clustering = clustering_statistics(G)
    print("\nAverage Clustering Coefficient:",clustering)

    print("\nGenerating Graph Features...")
    graph_features = (create_graph_feature_dataset(G))
    print("\nGraph Feature Dataset Shape:")
    print(graph_features.shape)
    print("\nSample Graph Features:")

    print(graph_features.head())

    final_df = processed_df.merge(
    graph_features,
    on="tx_id",
    how="left")

    print("\nFinal Dataset Shape:")
    print(final_df.shape)

    processed_dir = Path(
    "data/processed")

    processed_dir.mkdir(
    parents=True,
    exist_ok=True)

    final_df.to_csv(
    processed_dir /
    "graph_feature_dataset.csv",
    index=False)

    print("\nFinal Dataset Sample:")
    print(final_df.head())

    print("\nPreparing Dataset For Machine Learning...")
    ml_df = prepare_dataset(final_df)

    train_df, val_df, test_df = (
    temporal_split(ml_df))

    print("\nTemporal Split")

    print(f"Train Shape: {train_df.shape}")

    print(f"Validation Shape: {val_df.shape}")
    print(f"Test Shape: {test_df.shape}")

    print("\nML Dataset Shape:")
    print(ml_df.shape)

    print("\nTraining Logistic Regression...")

    model, scaler, results = (train_logistic_temporal(
    train_df,
    test_df
))

    Path("models").mkdir(exist_ok=True)
    save_model(model,scaler)

    print("\nModel Metrics")
    print("-" * 40)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    print("\nTraining XGBoost...")
    xgb_model, xgb_results, feature_names = (train_xgboost_temporal(
    train_df,
    test_df
))
    save_xgboost(xgb_model)

    print("\nXGBoost Metrics")
    print("-" * 40)
    for metric, value in (xgb_results.items()):
        print(f"{metric}: {value:.4f}")

    plot_feature_importance(xgb_model,feature_names) 


    print("\nPreparing GraphSAGE Data...")
    graph_data = build_graph_data(processed_df,edges)  

    graphsage_model = GraphSAGE(graph_data.num_node_features)

    print("\nTraining GraphSAGE...")
    graphsage_results = (train_graphsage(graphsage_model,graph_data,epochs=20))

    print("\nGraphSAGE Results")
    print("-" * 40)
    for k, v in (graphsage_results.items()):
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()