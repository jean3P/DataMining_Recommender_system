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
    features_with_edges_count = pd.merge(features, edges_count, how='left', left_on='Id', right_on='numeric_id_1')
    features_with_edges_count['num_edges_src'] = features_with_edges_count['num_edges_src'].fillna(0)

    # Counting target edges
    edges_count = edges.groupby('numeric_id_2').size().reset_index(name='num_edges_target')
    features_with_edges_count = pd.merge(features_with_edges_count, edges_count, how='left', left_on='Id', right_on='numeric_id_2')
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


def PopularityRecommender(node, features, edges):
    '''
        Returns a list of recommendations based on the popularity inside of the community
    '''

    # Ranking by Community
    comm_number = node.iloc[0]['Community']

    community = features.loc[features['Community']==comm_number].sort_values(by='num_edges', ascending=False)

    is_linked = False
    top_rank = pd.DataFrame(columns=community.columns)

    for index, row in community.iterrows():
        if IsLinked(edges, node.iloc[0]['Id'], row[0])==False:
            top_rank = top_rank.append(row, ignore_index=True)
            if top_rank.shape[0]==5:
                break
    
    print('user values:', node)
    print('recommendations', top_rank)
    
    

def main():

    # Loading data
    louvain = pd.read_csv(louvain_file)
    leiden = pd.read_csv(leiden_file)
    edges = pd.read_csv(large_twitch_edges)
    features = pd.read_csv(large_twitch_features)

    # Adding community to Node features
    features = features.rename(columns={'numeric_id': 'Node'})
    features_comm = pd.merge(leiden, features, on="Node")
    features_comm = features_comm.rename(columns={'Node': 'Id'})

    features_num_edges = NumberOfEdges(features_comm, edges)

    np.random.seed(0)
    node_to_recommend = features_comm.sample(1)

    PopularityRecommender(node_to_recommend, features_num_edges, edges)


if __name__ == "__main__":
    main()
