import os
import sys

abs_path = sys.path[0]

base_name = os.path.dirname(abs_path)

train_80 = os.path.join("./../resources/dataset_communities/train_80_features.csv")
test_20 = os.path.join("./../resources/dataset_communities/test_20_features.csv")
path_root_backend_classification_directory = os.path.join('./../../backend_rs/twitch_app/src/classification')
leiden_file = os.path.join("./../resources/leiden/leiden_community_assignments.csv")
large_twitch_edges = os.path.join("./../resources/twitch_gamers/large_twitch_edges.csv")
large_twitch_features = os.path.join("./../resources/twitch_gamers/large_twitch_features.csv")
trained_models_path = os.path.join("./../resources/models/")
communities_path = os.path.join("./../resources/communities/")
