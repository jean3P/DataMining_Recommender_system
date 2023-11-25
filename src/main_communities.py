import csv
import os.path

import leidenalg
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import logging
from dask import delayed
import igraph as ig
import community
import random
from constants import large_twitch_edges, large_twitch_features, outputs_path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@delayed
def compute_all_shortest_paths(G, node):
    logger.info(f"Computing shortest paths for node {node}")
    paths = nx.single_source_shortest_path_length(G, node)
    logger.info(f"Completed computing shortest paths for node {node}")
    return paths


def analyze_communities(G, partition):
    community_analysis = {}

    # Create a reverse mapping from community to nodes
    communities = {}
    for node, community in partition.items():
        if community not in communities:
            communities[community] = []
        communities[community].append(node)

    for community, nodes in communities.items():
        languages = {}
        mature_count = 0
        non_mature_count = 0

        for node in nodes:
            # Get node attributes
            attrs = G.nodes[node]

            # Count languages
            lang = attrs['language']
            if lang in languages:
                languages[lang] += 1
            else:
                languages[lang] = 1

            # Count mature and non-mature channels
            if attrs['mature'] == 1:
                mature_count += 1
            else:
                non_mature_count += 1

        # Check if there's more than one language in the community
        multiple_languages = len(languages) > 1

        # Check if there's more than one value for 'mature' in the community
        multiple_mature_values = mature_count > 0 and non_mature_count > 0

        community_analysis[community] = {
            'multiple_languages': multiple_languages,
            'languages': languages,
            'multiple_mature_values': multiple_mature_values,
            'mature_count': mature_count,
            'non_mature_count': non_mature_count
        }

    return community_analysis


def save_communities_to_csv(partition, filename, subset_filename):
    """Save community assignments to a CSV file."""

    # Save the entire partition
    filename_path = os.path.join(outputs_path, filename)
    subset_filename_path = os.path.join(outputs_path, subset_filename)
    with open(filename_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Node", "Community"])
        for node, community in partition.items():
            writer.writerow([node, community])

    # Shuffle the partition dictionary
    items = list(partition.items())
    random.shuffle(items)

    # Take 15% of the shuffled partition
    subset_size = int(len(items) * 0.001)
    subset_size = 400
    subset = dict(items[:subset_size])
    print(subset_size)

    # Save the 25% subset
    with open(subset_filename_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Node", "Community"])
        for node, community in subset.items():
            writer.writerow([node, community])


def internal_external_links_analysis(G, partition):
    community_edges = {}
    for (u, v) in G.edges():
        comm_u = partition[u]
        comm_v = partition[v]

        if comm_u == comm_v:
            # Internal link
            community_edges[comm_u] = community_edges.get(comm_u, {"internal": 0, "external": 0})
            community_edges[comm_u]["internal"] += 1
        else:
            # External link
            community_edges[comm_u] = community_edges.get(comm_u, {"internal": 0, "external": 0})
            community_edges[comm_u]["external"] += 1

            community_edges[comm_v] = community_edges.get(comm_v, {"internal": 0, "external": 0})
            community_edges[comm_v]["external"] += 1

    for comm, edges in community_edges.items():
        ratio = edges["internal"] / (edges["internal"] + edges["external"])
        print(f"    Community {comm}: Internal Links Ratio = {ratio:.2f}")


def louvain_community_detection(G):
    partition = community.best_partition(G)
    modularity_value = community.modularity(partition, G)
    print(f"==== LOUVAIN ====")
    print(f"    Modularity: {modularity_value}")
    save_communities_to_csv(partition, 'louvain_community_assignments.csv', 'louvain_community_subset.csv')
    internal_external_links_analysis(G, partition)
    community_analysis = analyze_communities(G, partition)
    for community_, data in community_analysis.items():
        print(f"Community {community_}:")
        print(f"    Languages: {data['languages']}")
        print(f"    Mature channels: {data['mature_count']}")
        print(f"    Non-mature channels: {data['non_mature_count']}")


def perform_leiden_community_detection(G):
    # Convert networkx graph to igraph
    ig_graph = ig.Graph.TupleList(G.edges(), directed=False)
    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
    leiden_partition = dict(zip(ig_graph.vs["name"], partition.membership))
    save_communities_to_csv(leiden_partition, 'leiden_community_assignments.csv', 'leiden_community_subset.csv')
    modularity_value = partition.modularity
    print(f"==== LEIDEN ====")
    print(f"    Modularity: {modularity_value}")
    internal_external_links_analysis(G, leiden_partition)
    community_analysis = analyze_communities(G, leiden_partition)
    for community_, data in community_analysis.items():
        print(f"Community {community_}:")
        print(f"    Languages: {data['languages']}")
        print(f"    Mature channels: {data['mature_count']}")
        print(f"    Non-mature channels: {data['non_mature_count']}")


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
