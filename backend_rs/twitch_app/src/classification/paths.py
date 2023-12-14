import os

train_80 = os.path.join("./../../resources/train_80_features.csv")
test_20 = os.path.join("./../../resources/test_20_features.csv")
resources_path = os.path.join("twitch_app/resources")
models_path_lp = os.path.join(resources_path, "models")
large_twitch_edges = os.path.join(resources_path, 'twitch_gamers', 'large_twitch_edges.csv')
log_file_4 = os.path.join(resources_path, "outputs", "link_prediction_recommender_log.txt")
leiden_file = os.path.join(resources_path, "outputs", "leiden_community_assignments.csv")
large_twitch_features = os.path.join(resources_path, 'twitch_gamers', 'large_twitch_features.csv')
log_file_2 = os.path.join(resources_path, "outputs", "LinkPredictor_log.txt")