import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from LinkPredictor import LinkPredictor 

import logging

from constants import log_file_4, large_twitch_edges, large_twitch_features, outputs_path, louvain_file, leiden_file

def configure_logging():
    logging.basicConfig(filename=log_file_4, level=logging.INFO, format="%(message)s", filemode='w')


# Set seed for reproducibility
np.random.seed(0)


def main():

    # Loading nodes with community and edges 
    nodes_leiden = pd.read_csv(leiden_file)
    edges = pd.read_csv(large_twitch_edges)

    edges = edges.rename(columns={'numeric_id_1': 'source', 'numeric_id_2': 'target'})

    # Split nodes by community
    communities = nodes_leiden.groupby('Community')
    community_nodes = {community: df for community, df in communities}

    # Group edges by community  
    community_edges = {}
    for community, nodes_leiden in community_nodes.items():
        # Get user IDs in the community
        nodes_ids = set(nodes_leiden['Node'])
        # Filter edges where both source and target are in user_ids
        edges_df = edges[edges['source'].isin(nodes_ids) & edges['target'].isin(nodes_ids)]
        community_edges[community] = edges_df

    # print(type(community_edges[1]))
    # print(community_edges)

    # Initialize the predictor
    predictor = LinkPredictor(community_nodes[3], community_edges[3])

    # Create graph and train models
    predictor.create_graph()
    predictor.train_node2vec()
    predictor.train_link_predictor()

    print('Training finished.')
    
    

    
if __name__ == "__main__":
    configure_logging()
    main()
    



