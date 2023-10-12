import csv

import leidenalg
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import logging
from dask import delayed
import igraph as ig
import community
import random
from constants import large_twitch_edges, large_twitch_features

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@delayed
def compute_all_shortest_paths(G, node):
    logger.info(f"Computing shortest paths for node {node}")
    paths = nx.single_source_shortest_path_length(G, node)
    logger.info(f"Completed computing shortest paths for node {node}")
    return paths


def save_communities_to_csv(partition, filename, subset_filename):
    """Save community assignments to a CSV file."""

    # Save the entire partition
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Node", "Community"])
        for node, community in partition.items():
            writer.writerow([node, community])

    # Shuffle the partition dictionary
    items = list(partition.items())
    random.shuffle(items)

    # Take 15% of the shuffled partition
    subset_size = int(len(items) * 0.015)
    subset = dict(items[:subset_size])

    # Save the 25% subset
    with open(subset_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Node", "Community"])
        for node, community in subset.items():
            writer.writerow([node, community])


def louvain_community_detection(G):
    partition = community.best_partition(G)
    save_communities_to_csv(partition, 'louvain_community_assignments.csv', 'louvain_community_subset.csv')


def perform_leiden_community_detection(G):
    # Convert networkx graph to igraph
    ig_graph = ig.Graph.TupleList(G.edges(), directed=False)
    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
    leiden_partition = dict(zip(ig_graph.vs["name"], partition.membership))
    save_communities_to_csv(leiden_partition, 'leiden_community_assignments.csv', 'leiden_community_subset.csv')


def main():
    edges = pd.read_csv(large_twitch_edges)
    features = pd.read_csv(large_twitch_features)

    # Construct a graph using NetworkX:
    G = nx.from_pandas_edgelist(edges, 'numeric_id_1', 'numeric_id_2')

    # Add node attributes from the features dataset:
    attribute_dict = features.set_index('numeric_id').to_dict('index')
    nx.set_node_attributes(G, attribute_dict)

    # Draw a subgraph for illustration
    sub_nodes = list(G.nodes())[0:10]
    subgraph = G.subgraph(sub_nodes)
    nx.draw(subgraph, with_labels=True, font_weight='bold')
    plt.show()

    # Basic Graph Properties
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Is the graph connected? {nx.is_connected(G)}")

    # Average Degree
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    print(f"Average degree: {avg_degree}")

    # Network Density
    density = nx.density(G)
    print(f"Density: {density}")

    # Execute community detection algorithms
    louvain_community_detection(G)
    # surprise_community_detection(G)
    perform_leiden_community_detection(G)


if __name__ == '__main__':
    main()
