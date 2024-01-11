import pandas as pd
import networkx as nx

from constants import large_twitch_edges

# Step 1: Read the CSV file
df = pd.read_csv(large_twitch_edges)

# Step 2: Create a directed graph
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row['numeric_id_1'], row['numeric_id_2'])

# Step 3: Check for reciprocal relationships
follows_back = []
for edge in G.edges():
    if G.has_edge(edge[1], edge[0]):
        follows_back.append((edge, True))
    else:
        follows_back.append((edge, False))

# Step 4: Output the results
for edge, has_reciprocal in follows_back:
    print(f"Edge {edge} has reciprocal relationship: {has_reciprocal}")
