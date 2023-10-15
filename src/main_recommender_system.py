import os.path
import pandas as pd
from constants import large_twitch_features, outputs_path, louvain_file, leiden_file


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


def get_algorithm_data(algorithm):
    if algorithm == 'louvain':
        return 'community_label_louvain', community_mature_preference_louvain
    elif algorithm == 'leiden':
        return 'community_label_leiden', community_mature_preference_leiden
    else:
        raise ValueError("Invalid algorithm choice. Choose either 'louvain' or 'leiden'.")


def recommend_content(features_df, user_numeric_id, algorithm='louvain'):
    """
    Generate content recommendations based on the user and community preferences.

    Parameters:
    - features_df (pd.DataFrame): DataFrame containing user features and community labels.
    - user_numeric_id (int): The numeric ID of the user to generate recommendations for.
    - algorithm (str): The algorithm used for community detection ('louvain' or 'leiden').

    Returns:
    - str: A recommendation message.
    """

    # Get the appropriate community label column and mature preference based on the chosen algorithm
    community_label_column, community_mature_preference = get_algorithm_data(algorithm)

    # Retrieve the user data from the features dataframe using the provided user_numeric_id
    user_data = features_df.loc[features_df['numeric_id'] == user_numeric_id]

    # Extract the user's community label and mature preference value
    user_community = user_data[community_label_column].values[0]
    user_mature_pref = user_data['mature'].values[0]

    # Extract the mature preference value for the community the user belongs to
    community_pref = community_mature_preference[user_community]

    # Generate recommendations based on the user's and community's mature content preference
    if user_mature_pref == 1:
        if community_pref > 0.5:
            return f"[{algorithm.capitalize()}-based Algorithm] Recommend Mature Content"
        else:
            return f"[{algorithm.capitalize()}-based Algorithm] Recommend Other Content"
    else:
        if community_pref <= 0.5:
            return f"[{algorithm.capitalize()}-based Algorithm] Recommend Other Content"
        else:
            return f"[{algorithm.capitalize()}-based Algorithm] Recommend Mature Content"


def hybrid_recommendation(features_df, user_numeric_id, algorithm='louvain'):
    """
    Generate a hybrid content recommendation based on both individual user preferences and community preferences.

    Parameters:
    - features_df (pd.DataFrame): DataFrame containing user features and community labels.
    - user_numeric_id (int): The numeric ID of the user to generate recommendations for.
    - algorithm (str): The algorithm used for community detection ('louvain' or 'leiden').

    Returns:
    - str: A recommendation message based on the hybrid approach.
    """

    # Get the appropriate community label column and mature preference based on the chosen algorithm
    community_label_column, community_mature_preference = get_algorithm_data(algorithm)

    # Extract the user's community label and mature preference from the features dataframe
    user_community = features_df.loc[features_df['numeric_id'] == user_numeric_id, community_label_column].values[0]
    user_mature_pref = features_df.loc[features_df['numeric_id'] == user_numeric_id, 'mature'].values[0]

    # If both user and community primarily prefer mature content, recommend mature content
    if user_mature_pref == 1 and community_mature_preference[user_community] > 0.5:
        return f"[{algorithm.capitalize()}-Hybrid Algorithm] Recommend Mature Content"

    # If both user and community primarily prefer non-mature content, recommend non-mature content
    elif user_mature_pref == 0 and community_mature_preference[user_community] <= 0.5:
        return f"[{algorithm.capitalize()}-Hybrid Algorithm] Recommend Other Content"

    else:
        # If there's a disagreement between user's and community's preferences, prioritize the user's preference
        # This ensures personalized recommendations even if the community's overall preference is different
        return f"[{algorithm.capitalize()}-Hybrid Algorithm] Recommend Mature Content" if user_mature_pref == 1 else (
            f"[{algorithm.capitalize()}-Hybrid Algorithm] Recommend Other Content")


dataframe = load_data()

# Calculate community preferences
community_mature_preference_louvain = dataframe.groupby('community_label_louvain')['mature'].mean()
community_mature_preference_leiden = dataframe.groupby('community_label_leiden')['mature'].mean()

user_id_to_recommend = 123456

recommendation_louvain = recommend_content(dataframe, user_id_to_recommend, 'louvain')
print(recommendation_louvain)

recommendation_leiden = recommend_content(dataframe, user_id_to_recommend, 'leiden')
print(recommendation_leiden)

hybrid_rec_louvain = hybrid_recommendation(dataframe, user_id_to_recommend, 'louvain')
print(hybrid_rec_louvain)

hybrid_rec_leiden = hybrid_recommendation(dataframe, user_id_to_recommend, 'leiden')
print(hybrid_rec_leiden)
