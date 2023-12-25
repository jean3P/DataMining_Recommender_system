import os
import sys

abs_path = sys.path[0]

base_name = os.path.dirname(abs_path)

train_80 = os.path.join("./../resources/dataset_communities/train_80_features.csv")
test_20 = os.path.join("./../resources/dataset_communities/test_20_features.csv")
path_root_backend_classification_directory = os.path.join('./../../backend_rs/twitch_app/src/classification')