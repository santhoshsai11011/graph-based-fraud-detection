import networkx as nx
from pyvis.network import Network


def build_transaction_network(
    transaction_id,
    edges_df,
    dataset,
    output_file="temp_graph.html"
):

    G = nx.Graph()

    outgoing = edges_df[
        edges_df["txId1"] == transaction_id
    ]

    incoming = edges_df[
        edges_df["txId2"] == transaction_id
    ]

    neighbors = set(
        outgoing["txId2"]
    )

    neighbors.update(
        incoming["txId1"]
    )

    print(f"Outgoing Connections: {len(outgoing)}")
 
    print(f"Incoming Connections: {len(incoming)}")

    print(f"Total Neighbors: {len(neighbors)}")




    G.add_node(transaction_id)

    for neighbor in neighbors:

        G.add_node(neighbor)

        G.add_edge(
            transaction_id,
            neighbor
        )

    net = Network(
        height="700px",
        width="100%",
        bgcolor="#111111",
        font_color="white"
    )

    def get_label(tx_id):

        row = dataset[
            dataset["tx_id"] == tx_id
        ]

        if row.empty:
            return "UNKNOWN"

        value = str(
            row["class"].iloc[0]
        )

        if value == "1":
            return "FRAUD"

        elif value == "2":
            return "LEGITIMATE"

        return "UNKNOWN"

    for node in G.nodes():

        label = get_label(node)

        color = "yellow"

        if label == "FRAUD":
            color = "red"

        elif label == "LEGITIMATE":
            color = "green"

        size = 15

        if node == transaction_id:
            color = "blue"
            size = 35

        net.add_node(
            node,
            label=str(node),
            title=f"Transaction ID: {node}<br>Status: {label}",
            color=color,
            size=size
        )

    for source, target in G.edges():
        net.add_edge(
            source,
            target
        )

    net.save_graph(
        output_file
    )

    return output_file