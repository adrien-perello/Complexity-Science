"""Helper functions for
- network metrics
- generating various random graph given an expected average node degree.

From:
https://github.com/adrien-perello/Computer-Science-Crash-Course/tree/main/06_Scientific-Computing-Libraries/Networkx/scripts
"""

import networkx as nx
import numpy as np


def avg_degree(nx_graph):
    """Return the average node degree of a graph

    Args:
        nx_graph (nx graph): a networkx graph

    Returns:
        float: average node degree
    """
    return 2 * nx_graph.number_of_edges() / nx_graph.number_of_nodes()


def degree_distribution(nx_graph):
    """Return the degree distribution of a graph

    Args:
        nx_graph (nx graph): a networkx graph

    Returns:
        ndarray[int]: number of nodes with i edges (i= index)
    """
    return np.array(nx.degree_histogram(nx_graph))


def nb_components(nx_graph):
    """Return the number of components of the graph

    Args:
        nx_graph (nx graph): a networkx graph

    Returns:
        int: number of components
    """
    return len(list(nx.connected_components(nx_graph)))


def avg_distance(nx_graph):
    """Return the average distance of each component of the graph

    Args:
        nx_graph (nx graph): a networkx graph

    Returns:
        ndarray[float]: average distance of each component
    """
    return np.array(
        [
            nx.average_shortest_path_length(subgraph)
            for subgraph in nx.connected_components(nx_graph)
        ]
    )


def network_metrics(nx_graph):
    """Returns common network metrics

    Args:
        nx_graph (nx graph): a networkx graph

    Returns:
        dict(
            nb components (int):   number of components
            avg degree (float):    average degree
            avg distance (float):  average distance
            degree distribution (ndarray[int]): degree distribution
        )
    """
    return {
        "nb components": nb_components(nx_graph),
        "avg degree": np.round(np.mean(avg_degree(nx_graph)), 2),
        "avg distance": np.round(avg_distance(nx_graph), 4),
        "degree distribution": degree_distribution(nx_graph),
    }


def avg_degree_to_erdos_renyi_kwargs(nb_nodes, avg_deg):
    """Convert parameters to usable inputs for networkx.erdos_renyi_graph()

    Args:
        nb_nodes (int):     number of nodes
        avg_deg (float): expected average degree of the graph

    Returns:
        dict(          keyword arguments
            n (int):   number of nodes
            p (float): edge probability (in [0,1])
        )
    """
    edge_probability = avg_deg / nb_nodes
    return {"n": nb_nodes, "p": edge_probability}


def avg_degree_to_watts_strogatz_kwargs(nb_nodes, avg_deg, rewiring_probability):
    """Convert parameters to usable inputs for networkx.watts_strogatz_graph()

    Args:
        nb_nodes (int):               number of nodes
        avg_deg (float):           expected average degree of the graph
        rewiring_probability (float): rewiring probability (in [0,1])

    Returns:
        dict(          keyword arguments
            n (int):   number of nodes
            k (int):   initial number of neighbors
            p (float): rewiring probability (in [0,1])
        )
    """
    nb_neighbours = round(avg_deg)
    return {"n": nb_nodes, "k": nb_neighbours, "p": rewiring_probability}


def avg_degree_to_barabasi_albert_kwargs(nb_nodes, avg_deg):
    """Convert parameters to usable inputs for networkx.barabasi_albert_graph()

    Args:
        nb_nodes (int):     number of nodes
        avg_deg (float): expected average degree of the graph

    Returns:
        dict(          keyword arguments
            n (int):   number of nodes
            m (int):   initial clique
        )
    """
    if not compatible_barabasi_albert_params(nb_nodes, avg_deg):
        raise ValueError(
            f"barabasi_albert_graph():"
            f"expected average degree ({avg_deg}) is too low"
            f"compared to the number of nodes ({nb_nodes})."
        )
    nb_neighbours = get_barabasi_albert_m_arg_from_avg_degree(nb_nodes, avg_deg)
    return {"n": nb_nodes, "m": nb_neighbours}


def compatible_barabasi_albert_params(nb_nodes, avg_deg):
    """Check if necessary conditions are met for generating a Barabasi Albert graph.
    See annex (end of notebook) for more details about the explanation.

    Args:
        nb_nodes (int):             number of nodes
        avg_deg (float):  expected average degree of the graph

    Returns:
        bool: valid parameters
    """
    return avg_deg <= nb_nodes ** 2 / (2 * (nb_nodes + 1))


def get_barabasi_albert_m_arg_from_avg_degree(nb_nodes, avg_deg):
    """Generate the initial clique value (m) inputs for networkx.barabasi_albert_graph().
    See annex (end of notebook) for more details about the explanation.

    Args:
        n (int):            number of nodes
        avg_deg (float): expected average degree of the graph

    Returns:
        int:   initial clique (m)
    """
    return round(
        np.min(np.roots([2 / (nb_nodes + 1), 2 / (nb_nodes + 1) - 2, avg_deg]))
    )


def generate_random_graph(network_type, nb_nodes, avg_deg):
    """Generate a random graph

    Args:
        network_type (string): type of random graph (erdos_renyi / watts_strogatz / barabasi_albert)
        nb_nodes (int): number of nodes
        avg_deg (int/float): expected average degree

    Returns:
        networkx.classes.graph.Graph: a networkx graph
    """
    graph = {
        "random": {
            "generator": nx.erdos_renyi_graph,
            "param": avg_degree_to_erdos_renyi_kwargs(nb_nodes, avg_deg),
        },
        "small world": {
            "generator": nx.watts_strogatz_graph,
            "param": avg_degree_to_watts_strogatz_kwargs(nb_nodes, avg_deg, 0.5),
        },
        "scale free": {
            "generator": nx.barabasi_albert_graph,
            "param": avg_degree_to_barabasi_albert_kwargs(nb_nodes, avg_deg),
        },
    }
    return graph[network_type]["generator"](**graph[network_type]["param"])
