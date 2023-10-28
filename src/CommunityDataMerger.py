import os.path
import pandas as pd
from constants import large_twitch_features, outputs_path, louvain_file, leiden_file, features_with_communities


def load_data():
    # Load the main features data and the community assignments from both algorithms
    features_df = pd.read_csv(large_twitch_features)
    louvain_communities = pd.read_csv(louvain_file)
    leiden_communities = pd.read_csv(leiden_file)

    # Merge Louvain community data
    features_df = pd.merge(features_df, louvain_communities[['Node', 'Community']],
                           left_on='numeric_id', right_on='Node', how='left')
    features_df.drop(columns=['Node'], inplace=True)
    features_df.rename(columns={'Community': 'community_label_louvain'}, inplace=True)

    # Merge Leiden community data
    features_df = pd.merge(features_df, leiden_communities[['Node', 'Community']],
                           left_on='numeric_id', right_on='Node', how='left')
    features_df.drop(columns=['Node'], inplace=True)
    features_df.rename(columns={'Community': 'community_label_leiden'}, inplace=True)

    # Save the merged data to a CSV file
    filename_path = os.path.join(outputs_path, 'large_twitch_features_with_communities.csv')
    features_df.to_csv(filename_path, index=False)

    return features_df


dataframe = load_data()

dataframe.to_csv(features_with_communities, index=False)
