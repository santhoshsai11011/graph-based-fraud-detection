import networkx as nx


def build_transaction_graph(edges_df):
    G = nx.from_pandas_edgelist(
        edges_df,
        source="txId1",
        target="txId2",
        create_using=nx.DiGraph()
    )

    return G