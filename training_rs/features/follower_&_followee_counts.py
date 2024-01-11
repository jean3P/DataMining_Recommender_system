import pandas as pd
import networkx as nx

from constants import large_twitch_edges

# Step 1: Read the CSV file
df = pd.read_csv(large_twitch_edges)

# Step 2: Create a directed graph
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row['numeric_id_1'], row['numeric_id_2'])

# Step 3: Calculate follower and followee counts
follower_counts = {node: G.in_degree(node) for node in G.nodes()}
followee_counts = {node: G.out_degree(node) for node in G.nodes()}

# Step 4: Output the results
for node in G.nodes():
    print(f"Node {node} has {follower_counts[node]} followers and {followee_counts[node]} followees")

# Convert follower and followee count dictionaries to DataFrames
df_followers = pd.DataFrame.from_dict(follower_counts, orient='index', columns=['Followers'])
df_followees = pd.DataFrame.from_dict(followee_counts, orient='index', columns=['Followees'])
