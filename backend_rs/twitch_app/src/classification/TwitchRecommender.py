import json
import pandas as pd
import numpy as np
import joblib
import logging
import os
from twitch_app.src.classification.paths import log_file_4, models_path_lp, leiden_file, large_twitch_edges, large_twitch_features
from twitch_app.src.classification.popularity_recommender import PopularityRecommender


class TwitchRecommenderSystem:
    """
    Class representing a Twitch Recommender System.

    Methods:
        - configure_logging()
            Configures logging for the system.

        - twitch_link_predict_recommender(new_user_id, new_user_community, community_nodes, community_edges)
            Performs link prediction and returns recommendations for a given user in a specific community.

            Parameters:
                - new_user_id (int): The ID of the new user.
                - new_user_community (int): The ID of the community to which the new user belongs.
                - community_nodes (dict): Dictionary containing community ID as key and corresponding nodes as value.
                - community_edges (dict): Dictionary containing community ID as key and corresponding edges as value.

            Returns:
                - link_predictions (DataFrame): DataFrame containing link predictions and scores, sorted by link probability.

        - twitch_recommender(new_user_id, new_user_community)
            Recommends Twitch channels to a new user in a specific community.

            Parameters:
                - new_user_id (int): The ID of the new user.
                - new_user_community (int): The ID of the community to which the new user belongs.

            Returns:
                - json_recommendations (str): JSON string containing the recommendations for the new user.

        - main()
            Entry point for the Twitch Recommender System. Runs example queries and prints the recommendations.
    """
    def __init__(self):
        # Set seed for reproducibility
        np.random.seed(0)
        self.configure_logging()

    @staticmethod
    def configure_logging():
        logging.basicConfig(filename=log_file_4, level=logging.INFO, format="%(message)s", filemode='w')

    def twitch_link_predict_recommender(self, new_user_id, new_user_community, community_nodes, community_edges):
        new_user_community_nodes = community_nodes[new_user_community]
        print(os.getcwd())
        model_filename = os.path.join(models_path_lp, 'link_prediction_community_' + str(new_user_community) + '.pkl')
        predictor = joblib.load(model_filename)

        link_predictions = pd.DataFrame(columns=new_user_community_nodes.columns)
        link_predictions['LinkProbability'] = None
        link_predictions['LinkPredScore'] = None

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

    def twitch_recommender(self, new_user_id, new_user_community):
        new_user_id = int(new_user_id)
        nodes_leiden = pd.read_csv(leiden_file)
        edges = pd.read_csv(large_twitch_edges)
        edges = edges.rename(columns={'numeric_id_1': 'source', 'numeric_id_2': 'target'})

        communities = nodes_leiden.groupby('Community')
        community_nodes = {community: df for community, df in communities}
        community_edges = {}

        for community, nodes in community_nodes.items():
            nodes_ids = set(nodes['Node'])
            edges_df = edges[edges['source'].isin(nodes_ids) & edges['target'].isin(nodes_ids)]
            community_edges[community] = edges_df

        if nodes_leiden['Node'].isin([new_user_id]).any():
            new_user_community = nodes_leiden.loc[nodes_leiden['Node'] == new_user_id, 'Community'].values[0]
            recommendations = self.twitch_link_predict_recommender(new_user_id, new_user_community, community_nodes, community_edges)
            features = pd.read_csv(large_twitch_features)
            features = features.rename(columns={'numeric_id': 'Node'})
            recommendations = pd.merge(recommendations, features, on="Node")
            recommendations["affiliate"] = recommendations["affiliate"].astype(bool)
            recommendations["mature"] = recommendations["mature"].astype(bool)
            recommendations["Community_rs"] = new_user_community

            json_recommendations = {
                "algorithm": "Link prediction",
                "results": recommendations[["Node", "affiliate", "language", "mature", "created_at", "updated_at", "LinkProbability", "LinkPredScore", "Community_rs"]].head(5).to_dict(orient='records')
            }
        else:
            recommendations = PopularityRecommender(new_user_id, new_user_community)
            recommendations["affiliate"] = recommendations["affiliate"].astype(bool)
            recommendations["mature"] = recommendations["mature"].astype(bool)
            json_recommendations = {
                "algorithm": "Popularity",
                "results": recommendations[["Node", "affiliate", "language", "mature", "created_at", "updated_at", "num_edges"]].head(5).to_dict(orient='records')
            }

        return json.dumps(json_recommendations)

    @staticmethod
    def main():
        recommender_system = TwitchRecommenderSystem()

        new_user_id = 37182
        new_user_community = 0
        print("Preparing list of recommendations for user: ", new_user_id)
        print(recommender_system.twitch_recommender(new_user_id, new_user_community))

        new_user_id = 170000
        new_user_community = 7
        print("Preparing list of recommendations for user: ", new_user_id)
        print(recommender_system.twitch_recommender(new_user_id, new_user_community))


if __name__ == "__main__":
    TwitchRecommenderSystem.main()
