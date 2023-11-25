import os
import sys

abs_path = sys.path[0]

base_name = os.path.dirname(abs_path)
resources_path = os.path.join(base_name, "resources/")

large_twitch_edges = os.path.join(resources_path, 'twitch_gamers', 'large_twitch_edges.csv')
large_twitch_features = os.path.join(resources_path, 'twitch_gamers', 'large_twitch_features.csv')

outputs_path = os.path.join(resources_path, "outputs")

louvain_file = os.path.join(outputs_path, "louvain_community_assignments.csv")
leiden_file = os.path.join(outputs_path, "leiden_community_assignments.csv")

train_df = os.path.join(outputs_path, "train_features.csv")
valid_df = os.path.join(outputs_path, "valid_features.csv")
test_df = os.path.join(outputs_path, "test_features.csv")
log_file_1 = os.path.join(outputs_path, "recommendation_log.txt")
train_80_df = os.path.join(outputs_path, "train_80_features.csv")
test_20_df = os.path.join(outputs_path, "test_20_features.csv")
features_with_communities = os.path.join(outputs_path, "features_with_communities.csv")

louvain_same_communities_png = os.path.join(outputs_path, "louvain_same_communities.png")
leiden_same_communities_png = os.path.join(outputs_path, "leiden_same_communities.png")

louvain_not_same_communities_png = os.path.join(outputs_path, "louvain_not_same_communities.png")
leiden_not_same_communities_png = os.path.join(outputs_path, "leiden_not_same_communities.png")
no_community_save_png = os.path.join(outputs_path, "no_community_save.png")

log_file_2 = os.path.join(outputs_path, "link_prediction_log.txt")
log_file_3 = os.path.join(outputs_path, "new-user_features_log.txt")