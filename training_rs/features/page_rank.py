import pandas as pd
import networkx as nx

from constants import large_twitch_edges

# 1. Read the file into a Pandas DataFrame
df = pd.read_csv(large_twitch_edges)

# 2. Create a graph from the DataFrame
G = nx.from_pandas_edgelist(df, 'numeric_id_1', 'numeric_id_2')

# 3. Use the PageRank algorithm to calculate the Page Rank
page_rank = nx.pagerank(G)

# 4. Print the PageRank for each node
for node, rank in page_rank.items():
    print(f'Node: {node}, Rank: {rank}')

# Convert the page_rank dictionary to a DataFrame
df_rank = pd.DataFrame.from_dict(page_rank, orient='index', columns=['PageRank'])