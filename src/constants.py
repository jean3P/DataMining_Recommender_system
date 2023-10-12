import os
import sys

abs_path = sys.path[0]

base_name = os.path.dirname(abs_path)
resources_path = os.path.join(base_name, "resources/")

large_twitch_edges = os.path.join(resources_path, 'twitch_gamers', 'large_twitch_edges.csv')
large_twitch_features = os.path.join(resources_path, 'twitch_gamers', 'large_twitch_features.csv')

