import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from pathlib import Path

import joblib

from popularity_recommender import PopularityRecommender

import logging
import os
from constants import log_file_4, large_twitch_edges, large_twitch_features, outputs_path, louvain_file, leiden_file, trained_models_path

def configure_logging():
    logging.basicConfig(filename=log_file_4, level=logging.INFO, format="%(message)s", filemode='w')


# Set seed for reproducibility
np.random.seed(0)


def TwitchLinkPredictRecommender(new_user_id, new_user_community, community_nodes, community_edges):

    # new_user_id = 37182
    # new_user_community = 7
    new_user_community_nodes = community_nodes[new_user_community]
    new_user_community_edges = community_edges[new_user_community]

    # Load the model from the file
    model_filename = os.path.join(trained_models_path, 'link_prediction_community_'+str(new_user_community)+'.pkl')
    predictor = joblib.load(model_filename)

    # link_predictions = pd.DataFrame(columns=['NewUserID', 'Node', 'LinkProbability', 'LinkPredScore'])
    link_predictions = pd.DataFrame(columns=new_user_community_nodes.columns)
    link_predictions['LinkProbability'] = None
    link_predictions['LinkPredScore'] = None

    # Calculate link probability and prediction for between the new user and the nodes in the community.
    for index, row in new_user_community_nodes.iterrows():
        node = row['Node']
        if node != new_user_id:
            prediction_score = predictor.predict_link(new_user_id, node)
            link_probability = predictor.predict_probability(new_user_id, node)

            new_row = row
            new_row['LinkProbability'] = link_probability
            new_row['LinkPredScore'] = prediction_score
            link_predictions.loc[len(link_predictions)] = new_row

    return link_predictions.sort_values(by='LinkProbability', ascending=False)


def TwitchRecommender(new_user_id, new_user_community):
    """
    :param new_user_id: Integer. User ID.
    :param new_user_community: Integer. Community ID.  
    :return: json con campos seleccionados y campo tipo de recommender
    """
    # Loading nodes with community and edges 
    nodes_leiden = pd.read_csv(leiden_file)
    edges = pd.read_csv(large_twitch_edges)

    edges = edges.rename(columns={'numeric_id_1': 'source', 'numeric_id_2': 'target'})

    # Split nodes by community
    communities = nodes_leiden.groupby('Community')
    community_nodes = {community: df for community, df in communities}

    # Group edges by community  
    community_edges = {}
    for community, nodes in community_nodes.items():
        # Get user IDs in the community
        nodes_ids = set(nodes['Node'])
        # Filter edges where both source and target are in user_ids
        edges_df = edges[edges['source'].isin(nodes_ids) & edges['target'].isin(nodes_ids)]
        community_edges[community] = edges_df

    if new_user_id in nodes_leiden['Node'].values:
        new_user_community = nodes_leiden.loc[nodes_leiden['Node']==new_user_id, 'Community'].values[0]
        print("User founded in dataset. Generating recommendations based on Link predictions...")
        recommendations = TwitchLinkPredictRecommender(new_user_id, new_user_community, community_nodes, community_edges)

        # Adding features to the recommendations
        features = pd.read_csv(large_twitch_features)
        features = features.rename(columns={'numeric_id': 'Node'})
        recommendations = pd.merge(recommendations, features, on="Node")
        recommendations["affiliate"] = recommendations["affiliate"].astype(bool)
        recommendations["mature"] = recommendations["mature"].astype(bool)
        recommendations["Community"] = new_user_community

        # Formatting into a json
        json_recommendations = {"algorithm": "Link prediction"}
        json_recommendations["recommendations"] = recommendations[["Node", "affiliate", "language", "mature", "created_at", "updated_at", "LinkProbability", "LinkPredScore", "Community"]].head(5).to_json(orient='records', lines=False)


    else:
        print("User not founded in dataset. Generating recommendations based on Popularity... ")
        recommendations = PopularityRecommender(new_user_id, new_user_community)
        recommendations["affiliate"] = recommendations["affiliate"].astype(bool)
        recommendations["mature"] = recommendations["mature"].astype(bool)

        # Formatting into a json
        json_recommendations = {"algorithm": "Popularity"}
        json_recommendations["recommendations"] = recommendations[["Node", "affiliate", "language", "mature", "created_at", "updated_at", "num_edges"]].head(5).to_json(orient='records', lines=False)

    return json_recommendations


def main():

    new_user_id = 37182
    new_user_community = 0

    print("Preparing list of recommendations for user: ", new_user_id)
    print(TwitchRecommender(new_user_id, new_user_community))

    # recomm = TwitchRecommender(new_user_id, new_user_community)
    # print(recomm.head(5))
    # print(recomm.head(5).to_json(orient='records', lines=False))


    new_user_id = 170000
    # new_user_id = 206981151
    new_user_community = 7
    # new_user_community = 1

    print("Preparing list of recommendations for user: ", new_user_id)
    print(TwitchRecommender(new_user_id, new_user_community))
    

if __name__ == "__main__":
    configure_logging()
    main()
    



