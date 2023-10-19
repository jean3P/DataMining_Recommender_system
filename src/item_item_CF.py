import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
from constants import large_twitch_edges, large_twitch_features

edges = pd.read_csv(large_twitch_edges)
# Randomly sample 60% of the dataset
edges = edges.sample(frac=0.6, random_state=42)

features = pd.read_csv(large_twitch_features)

# Convert the edges data into a sparse user-item matrix
# The edges data is converted into a sparse user-item matrix. This matrix indicates which user interacted with which
# item. The value is 1 if there's an interaction and 0 otherwise. Sparse matrices are memory-efficient ways to store
# large matrices with many zeros.
data = [1] * len(edges)  # Assuming a binary interaction (1 for interaction)
rows = edges['numeric_id_1'].values
cols = edges['numeric_id_2'].values

sparse_user_item = coo_matrix((data, (rows, cols)))


def compute_item_similarities(sparse_user_item):
    # Compute item-item cosine similarity
    item_similarity = cosine_similarity(sparse_user_item.T)
    item_similarity_df = pd.DataFrame(item_similarity)
    return item_similarity_df


def recommend_items(user, sparse_user_item, item_similarity_df, N=10):
    # Get items interacted by the user
    user_interactions = sparse_user_item.getrow(user).toarray().ravel()
    interacted_items = [i for i, interacted in enumerate(user_interactions) if interacted]

    # Get similar items and their scores
    similar_items_list = []
    for item in interacted_items:
        sim_items = item_similarity_df.iloc[item].drop(item)
        similar_items_list.append(sim_items)

    # Concatenate all the Series in the list
    similar_items = pd.concat(similar_items_list)

    # Aggregate and sort by similarity score
    recommendations = similar_items.groupby(similar_items.index).sum().sort_values(ascending=False)

    # Remove items already interacted with
    recommendations = recommendations[~recommendations.index.isin(interacted_items)]

    return recommendations.head(N)


def display_recommendations(recommended_items, features_df):
    header = "| No. |   Item ID  |  Views   | Mature | Life Time | Language | Similarity Score |"
    separator = "+" + "-" * 5 + "+" + "-" * 13 + "+" + "-" * 10 + "+" + "-" * 7 + "+" + "-" * 10 + "+" + "-" * 9 + "+" + "-" * 20 + "+"

    print(separator)
    print(header)
    print(separator)

    for idx, (item, score) in enumerate(recommended_items.items(), 1):
        item_features = features_df.loc[features_df['numeric_id'] == item]
        views = item_features['views'].iloc[0]
        mature = item_features['mature'].iloc[0]
        life_time = item_features['life_time'].iloc[0]
        language = item_features['language'].iloc[0]

        print(f"| {idx:3} | {item:10} | {views:8} | {mature:6} | {life_time:8} | {language:8} | {score:17.10f} |")

    print(separator)


item_similarity_df = compute_item_similarities(sparse_user_item)
user_sample = 98343  # Replace with any user ID from your dataset
recommended_items = recommend_items(user_sample, sparse_user_item, item_similarity_df)

display_recommendations(recommended_items, features)
