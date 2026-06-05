import networkx as nx
import pandas as pd


def compute_degree_features(G):
    return pd.DataFrame(
        {
            "tx_id": list(G.nodes()),
            "degree": [
                degree
                for _, degree in G.degree()
            ],
            "in_degree": [
                degree
                for _, degree in G.in_degree()
            ],
            "out_degree": [
                degree
                for _, degree in G.out_degree()
            ]
        }
    )


def compute_pagerank_feature(G):
    pagerank = nx.pagerank(
        G,
        alpha=0.85
    )

    return pd.DataFrame(
        {
            "tx_id": list(pagerank.keys()),
            "pagerank": list(pagerank.values())
        }
    )


def compute_clustering_feature(G):
    clustering = nx.clustering(
        G.to_undirected()
    )

    return pd.DataFrame(
        {
            "tx_id": list(clustering.keys()),
            "clustering_coef":
                list(clustering.values())
        }
    )


def create_graph_feature_dataset(G):

    degree_df = compute_degree_features(G)

    pagerank_df = compute_pagerank_feature(G)

    clustering_df = compute_clustering_feature(G)

    graph_features = (
        degree_df
        .merge(
            pagerank_df,
            on="tx_id"
        )
        .merge(
            clustering_df,
            on="tx_id"
        )
    )

    return graph_features