import os.path

import pandas as pd

from constants import large_twitch_edges, large_twitch_features, outputs_path, louvain_file, leiden_file

# Load the datasets into pandas dataframes
edges_df = pd.read_csv(large_twitch_edges)
features_df = pd.read_csv(large_twitch_features)

# Read the pre-calculated community assignments
louvain_communities = pd.read_csv(louvain_file)
print(louvain_communities.columns)

# Merge with the features_df dataframe
features_df = pd.merge(features_df, louvain_communities, left_on='numeric_id', right_on='Node', how='left')

# Now, 'features_df' will have a 'Community' column that contains community assignments
# If you want it to be named 'community_label', you can rename it
features_df.rename(columns={'Community': 'community_label'}, inplace=True)

filename = 'large_twitch_features_with_communities.csv'
filename_path = os.path.join(outputs_path, filename)
features_df.to_csv(filename_path, index=False)

# Calculate the proportion of users preferring mature content in each community
community_mature_preference = features_df.groupby('community_label')['mature'].mean()

print(community_mature_preference)


# User-based Recommendations
def recommend_content(user_numeric_id):
    user_community = features_df.loc[features_df['numeric_id'] == user_numeric_id, 'community_label'].values[0]

    if community_mature_preference[user_community] > 0.5:  # Change this threshold as per requirement
        return "[Community-based Algorithm] Recommend Mature Content"
    else:
        return "[Community-based Algorithm] Recommend Other Content"


user_id_to_recommend = 123456  # Replace with the actual numeric_id
recommendation = recommend_content(user_id_to_recommend)
print(recommendation)


#  Hybrid Strategy
def hybrid_recommendation(user_numeric_id):
    user_community = features_df.loc[features_df['numeric_id'] == user_numeric_id, 'community_label'].values[0]
    user_mature_pref = features_df.loc[features_df['numeric_id'] == user_numeric_id, 'mature'].values[0]

    # If both the user and the community predominantly prefer mature content
    if user_mature_pref == 1 and community_mature_preference[user_community] > 0.5:
        return "[Hybrid Algorithm] Recommend Mature Content"
    elif user_mature_pref == 0 and community_mature_preference[user_community] <= 0.5:
        return "[Hybrid Algorithm] Recommend Other Content"
    else:
        # Handle the scenario where user's preference and community preference doesn't align
        # Here, giving preference to user's individual preference
        return "[Hybrid Algorithm] Recommend Mature Content" if user_mature_pref == 1 else ("[Hybrid Algorithm] "
                                                                                            "Recommend Other Content")


hybrid_recommendation = hybrid_recommendation(user_id_to_recommend)
print(hybrid_recommendation)
