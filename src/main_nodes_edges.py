import pandas as pd
import numpy as np
import os.path
from constants import large_twitch_edges, large_twitch_features, outputs_path, louvain_file


# Loading datasets
louvain = pd.read_csv(louvain_file)
edges = pd.read_csv(large_twitch_edges)
features = pd.read_csv(large_twitch_features)

features = features.rename(columns={'numeric_id': 'Node'})

# Merge the subset community assignments with node features
merged_df = pd.merge(louvain, features, on="Node")
merged_df = merged_df.rename(columns={'Node': 'Id'})

# Set a seed value
np.random.seed(0)

# Select 100 random edges
subset_edges = edges.sample(n=150)

# Filtering nodes
subset_features = merged_df[merged_df['Id'].isin(subset_edges['numeric_id_1']) |
                           merged_df['Id'].isin(subset_edges['numeric_id_2'])
                           ]

subset_edges = subset_edges.rename(columns={'numeric_id_1': 'Source'})
subset_edges = subset_edges.rename(columns={'numeric_id_2': 'Target'})

# print(filtered_nodes.head())
subset_features_filename = os.path.join(outputs_path, 'features_louvain_subset.csv')
subset_features.to_csv(subset_features_filename, index=False)
subset_edges_filename = os.path.join(outputs_path, 'edges_louvain_subset.csv')
subset_edges.to_csv(subset_edges_filename, index=False)
