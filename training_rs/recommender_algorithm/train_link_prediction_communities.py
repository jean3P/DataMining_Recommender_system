import pandas as pd
import numpy as np
import joblib
from LinkPredictor import LinkPredictor
import os
from constants import large_twitch_edges, leiden_file, trained_models_path

# Set seed for reproducibility
np.random.seed(0)


def main():
    # Loading nodes with community and edges
    nodes_leiden = pd.read_csv(leiden_file)
    edges = pd.read_csv(large_twitch_edges)

    # Renaming columns
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

    # Creating a scores list
    scores_ls = []
    comm_ls = []

    # Training model for each community.
    for community in community_nodes.keys():
        # Initialize the predictor
        predictor = LinkPredictor(community_nodes[community], community_edges[community])

        # Create graph and train models
        predictor.create_graph()
        predictor.train_node2vec()
        predictor.train_link_predictor()
        print(f'Training model for community {community} finished.')

        # Get auc score of the model
        auc_score = predictor.get_auc_score()
        scores_ls.append(auc_score)
        comm_ls.append(community)

        # Saving model
        model_filename = os.path.join(trained_models_path, 'link_prediction_community_' + str(community) + '.pkl')
        joblib.dump(predictor, model_filename)

        print(f'Model for community {community} is saved.')

    # Saving model scores
    scores_df = pd.DataFrame({'Community': comm_ls, 'Score': scores_ls})
    scores_filename = os.path.join(trained_models_path, 'auc_scores.csv')
    scores_df.to_csv(scores_filename, index=False)

    print('All models have been trained.')


if __name__ == "__main__":
    main()
