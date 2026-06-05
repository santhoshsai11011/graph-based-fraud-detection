import networkx as nx
from pyvis.network import Network


def build_transaction_network(
    transaction_id,
    edges_df,
    dataset,
    output_file="temp_graph.html"
):

    # ------------------------------------------
    # Build Full Graph
    # ------------------------------------------

    full_graph = nx.from_pandas_edgelist(
        edges_df,
        source="txId1",
        target="txId2"
    )

    # ------------------------------------------
    # Get 2-Hop Neighborhood
    # ------------------------------------------

    if transaction_id not in full_graph:

        net = Network(
            height="700px",
            width="100%"
        )

        net.save_graph(
            output_file
        )

        return output_file

    nodes = {transaction_id}

    first_hop = set(
        full_graph.neighbors(
            transaction_id
        )
    )

    nodes.update(
        first_hop
    )

    for neighbor in first_hop:

        second_hop = set(
            full_graph.neighbors(
                neighbor
            )
        )

        nodes.update(
            second_hop
        )

    subgraph = (
        full_graph
        .subgraph(nodes)
        .copy()
    )

    print(
        f"Visualization Nodes: "
        f"{subgraph.number_of_nodes()}"
    )

    print(
        f"Visualization Edges: "
        f"{subgraph.number_of_edges()}"
    )

    # ------------------------------------------
    # Create PyVis Network
    # ------------------------------------------

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#111111",
        font_color="white"
    )

    net.from_nx(
        subgraph
    )

    # ------------------------------------------
    # Node Labels / Colors
    # ------------------------------------------

    for node in net.nodes:

        tx_id = int(
            node["id"]
        )

        row = dataset[
            dataset["tx_id"]
            == tx_id
        ]

        label = "UNKNOWN"

        if not row.empty:

            value = str(
                row["class"]
                .iloc[0]
            )

            if value == "1":

                label = "FRAUD"

            elif value == "2":

                label = (
                    "LEGITIMATE"
                )

        color = "yellow"

        if label == "FRAUD":

            color = "red"

        elif label == "LEGITIMATE":

            color = "green"

        node["color"] = color

        node["title"] = (
            f"Transaction ID: {tx_id}"
            f"<br>Status: {label}"
        )

        node["label"] = str(
            tx_id
        )

        node["size"] = 12

        if tx_id == transaction_id:

            node["color"] = (
                "blue"
            )

            node["size"] = 40

    # ------------------------------------------
    # Physics Settings
    # ------------------------------------------

    net.set_options(
        """
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -2000,
              "springLength": 120
            },
            "minVelocity": 0.75
          }
        }
        """
    )

    net.save_graph(
        output_file
    )

    return output_file