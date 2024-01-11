import pandas as pd
import networkx as nx

from constants import large_twitch_edges

# 1. Read the file into a Pandas DataFrame
df = pd.read_csv(large_twitch_edges)

# 2. Create a graph from the DataFrame
G = nx.from_pandas_edgelist(df, 'numeric_id_1', 'numeric_id_2')

# 3. Set nodes
source_node = 98343
target_node = 141493

try:
    shortest_path = nx.shortest_path(G, source=source_node, target=target_node)
    print("Shortest path between {} and {} is: {}".format(source_node, target_node, shortest_path))
except nx.NetworkXNoPath:
    print("No path exists between {} and {}".format(source_node, target_node))
