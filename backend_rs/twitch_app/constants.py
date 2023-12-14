import os

import sys

abs_path = sys.path[0]

base_name = os.path.dirname(abs_path)
model_path = os.path.join("twitch_app/src/classification/svm_model.joblib")
community_labels = [0, 1, 2, 3, 6, 13, 36, 38, 44, 53, 59, 80, 86, 91, 93, 95, 108, 114]

# resources_path = os.path.join(base_name, "backend_rs","twitch_app")
# models_path_lp = os.path.join(os.getcwd(),"twitch_app","resources","models")
# large_twitch_edges = os.path.join(resources_path, 'resources', 'twitch_gamers', 'large_twitch_edges.csv')
# log_file_4 = os.path.join(resources_path, 'resources', "outputs", "link_prediction_recommender_log.txt")
# leiden_file = os.path.join(resources_path, 'resources', "outputs", "leiden_community_assignments.csv")
# large_twitch_features = os.path.join(resources_path, 'resources', 'twitch_gamers', 'large_twitch_features.csv')
# log_file_2 = os.path.join(resources_path, 'resources', "outputs", "LinkPredictor_log.txt")