# import libraries
import pandas as pd
import numpy as np
import os.path
from constants import large_twitch_edges, large_twitch_features, outputs_path, louvain_file, leiden_file

def NumberOfEdges(features, edges) :

    '''
        Returns a dataframe with the number of edges by node.
    '''

    # Counting source edges
    edges_count = edges.groupby('numeric_id_1').size().reset_index(name='num_edges_src')
    features_with_edges_count = pd.merge(features, edges_count, how='left', left_on='Node', right_on='numeric_id_1')
    features_with_edges_count['num_edges_src'] = features_with_edges_count['num_edges_src'].fillna(0)

    # Counting target edges
    edges_count = edges.groupby('numeric_id_2').size().reset_index(name='num_edges_target')
    features_with_edges_count = pd.merge(features_with_edges_count, edges_count, how='left', left_on='Node', right_on='numeric_id_2')
    features_with_edges_count['num_edges_target'] = features_with_edges_count['num_edges_target'].fillna(0)

    # Summing up edges
    features_with_edges_count['num_edges'] = features_with_edges_count['num_edges_src'] + features_with_edges_count['num_edges_target']
    features_with_edges_count = features_with_edges_count.drop(['numeric_id_1', 'numeric_id_2', 'num_edges_src', 'num_edges_target'], axis=1)

    return features_with_edges_count


def IsLinked(edges, node1_id, node2_id):
    '''
        This function returns True if there is a link between two nodes.
    '''
    is_linked = False

    mask1 = (edges['numeric_id_1'] == node1_id) & (edges['numeric_id_2'] == node2_id)
    mask2 = (edges['numeric_id_1'] == node2_id) & (edges['numeric_id_2'] == node1_id)
    mask = mask1 | mask2
    # print(mask1)

    if mask.any():
        is_linked = True
    
    return is_linked


def PopularityRecommender(new_user_id, new_user_community): 
    '''
        Returns a top rank list of recommendations based on the user popularity inside of the community
    '''

    # Loading data
    # louvain = pd.read_csv(louvain_file)
    leiden = pd.read_csv(leiden_file)
    edges = pd.read_csv(large_twitch_edges)
    features = pd.read_csv(large_twitch_features)

    # Adding community to Node features
    features = features.rename(columns={'numeric_id': 'Node'})
    features_comm = pd.merge(leiden, features, on="Node")
    # features_comm = features_comm.rename(columns={'Node': 'Id'})

    # Adding number of edges to Node features
    features_num_edges = NumberOfEdges(features_comm, edges)

    # Ranking by Community
    comm_number = new_user_community

    community = features_num_edges.loc[features_num_edges['Community']==comm_number].sort_values(by='num_edges', ascending=False)

    is_linked = False
    top_rank = pd.DataFrame(columns=community.columns)

    for index, row in community.iterrows():
        if IsLinked(edges, new_user_id, row.iloc[0])==False:
            top_rank = pd.concat([top_rank, row.to_frame().T], ignore_index=True)
            if top_rank.shape[0]==10:
                break
    
    return top_rank
    # print('user values:')
    # print('user id:', new_user_id)
    # print('user community:', new_user_community)
    # print('recommendations: \n', top_rank)
    # print('community info: \n', community.describe())


# def AllCommunityRecommendations(features, edges):

#     # Selecting some nodes for each Leiden community
#     nodes_id_comm = [141493, 98343, 1679, 30061, 30293, 164528, 47048, 100109, 25310, 91680, 22970, 16162, 17553, 122816, 146294, 1942, 152300, 132852, 109249, 67761, 44630, 77002]
#     number_comm = list(range(0,22))

#     nodes = features[features['Id'].isin(nodes_id_comm)].sort_values(by='Community')

#     print(nodes.shape)

#     for index, row in nodes.iterrows():
#         node = pd.DataFrame([row])
#         PopularityRecommender(node, features, edges)
    
    

def main():

    # Loading data
    # louvain = pd.read_csv(louvain_file)
    leiden = pd.read_csv(leiden_file)
    edges = pd.read_csv(large_twitch_edges)
    features = pd.read_csv(large_twitch_features)

    # Adding community to Node features
    features = features.rename(columns={'numeric_id': 'Node'})
    features_comm = pd.merge(leiden, features, on="Node")
    # features_comm = features_comm.rename(columns={'Node': 'Id'})

    # Selecting a random node to recommend
    np.random.seed(0)
    node_to_recommend = features_comm.sample(1)

    print(node_to_recommend.iloc[0]['Node'])

    print(PopularityRecommender(node_to_recommend.iloc[0]['Node'], node_to_recommend.iloc[0]['Community']))

    # AllCommunityRecommendations(features_num_edges, edges)




if __name__ == "__main__":
    main()
