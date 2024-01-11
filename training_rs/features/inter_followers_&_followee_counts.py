import pandas as pd
import networkx as nx
from itertools import combinations
import multiprocessing as mp

from constants import large_twitch_edges


def calculate_inter_counts(pair: tuple):
    node1, node2 = pair
    followers1 = set(G.predecessors(node1))
    followers2 = set(G.predecessors(node2))
    followees1 = set(G.successors(node1))
    followees2 = set(G.successors(node2))

    inter_followers = len(followers1.intersection(followers2))
    inter_followees = len(followees1.intersection(followees2))

    return (node1, node2), {'inter_followers': inter_followers, 'inter_followees': inter_followees}


# Step 1: Read the CSV file
df = pd.read_csv(large_twitch_edges)

# Step 2: Create a directed graph
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row['numeric_id_1'], row['numeric_id_2'])

# Step 3: Calculate inter followers and followees
pool = mp.Pool(mp.cpu_count())
inter_counts = dict(pool.map(calculate_inter_counts, combinations(G.nodes(), 2)))
pool.close()

# Step 4: Output the results
for pair, counts in inter_counts.items():
    print(
        f"Pair {pair} has {counts['inter_followers']} inter followers and {counts['inter_followees']} inter followees")
