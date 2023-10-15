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
