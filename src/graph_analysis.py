import networkx as nx


def basic_graph_stats(G):

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G)
    }


def connected_component_stats(G):

    components = list(
        nx.weakly_connected_components(G)
    )

    largest_component = max(
        components,
        key=len
    )

    return {
        "num_components": len(components),
        "largest_component_size":
            len(largest_component)
    }


def degree_statistics(G):

    degrees = [
        degree
        for node, degree
        in G.degree()
    ]

    return {
        "avg_degree":
            sum(degrees)/len(degrees),

        "max_degree":
            max(degrees),

        "min_degree":
            min(degrees)
    }


def clustering_statistics(G):

    undirected = G.to_undirected()

    return nx.average_clustering(
        undirected
    )