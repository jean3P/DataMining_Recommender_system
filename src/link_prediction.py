import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import logging

from constants import log_file_2, large_twitch_edges, large_twitch_features, outputs_path, louvain_file, leiden_file


def configure_logging():
    logging.basicConfig(filename=log_file_2, level=logging.INFO, format="%(message)s", filemode='w')


class LinkPredictor:
    def __init__(self, train_nodes, train_edges):
        """
        Initialize the Link predictor with the training dataset.

        :param train_nodes: DataFrame containing the training nodes data.
        :param train_edges: DataFrame containing the training edges data.
        """

        self.nodes = train_nodes
        self.edges = train_edges
        self.G = None
        self.model = None



    def create_graph(self):
        """
        Create a graph from the training data.

        """

        # Creating a graph from the edges
        self.G = nx.from_pandas_edgelist(self.edges, 'source', 'target')


    def train_node2vec(self):
        """
        Trains node2vec model.

        """

        try:
            # Initialize node2vec model
            node2vec = Node2Vec(self.G, dimensions=64, walk_length=30, num_walks=200, workers=4)

            # Train node2vec model
            self.model = node2vec.fit(window=10, min_count=1, batch_words=4)

            # Retrieve node embeddings
            embeddings = np.array([self.model.wv.get_vector(str(node)) for node in self.G.nodes()])

        except self.G is None:
            logging.error("Need to create a graph.")
            
        
    
    def train_link_predictor(self):
        """
        Trains link predictor
        """

        try: 
            # Prepare the dataset for the link prediction
            # Generate positive examples
            positive_examples = self.edges.copy()

            # Generate negative examples
            negative_examples = []
            while len(negative_examples) < len(positive_examples):
                source = np.random.choice(self.G.nodes())
                target = np.random.choice(self.G.nodes())
                if source != target and not self.G.has_edge(source, target):
                    negative_examples.append([source, target])
            negative_examples = pd.DataFrame(negative_examples, columns=['source', 'target'])

            # Combine positive and negative examples
            examples = pd.concat([positive_examples, negative_examples], ignore_index=True)
            examples['label'] = [1] * len(positive_examples) + [0] * len(negative_examples)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                examples[['source', 'target']], examples['label'], test_size=0.3, random_state=42
            )

            # Apply prediction to the dataset
            X_train['similarity'] = [self.predict_link(row['source'], row['target']) for index, row in X_train.iterrows()]

            # Train a logistic regression model
            lr = LogisticRegression(random_state=42)
            lr.fit(X_train[['similarity']], y_train)

            # Predict on the test set
            X_test['similarity'] = [self.predict_link(row['source'], row['target']) for index, row in X_test.iterrows()]
            y_pred = lr.predict_proba(X_test[['similarity']])[:, 1]

            # Evaluate the model
            auc_score = roc_auc_score(y_test, y_pred)
            print(f'The AUC score of the link prediction model is: {auc_score}')

        
        except self.G is None:
            logging.error("Need to create a graph.")

        except self.model is None:
            logging.error("Need to train model.")

    
    def predict_link(self, source, target):
        try:
            vector_1 = self.model.wv.get_vector(str(source))
            vector_2 = self.model.wv.get_vector(str(target))

            return np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
        
        except self.model is None:
            logging.error("Need to train model.")


    
def main():

    # nodes = pd.read_csv(large_twitch_features)
    # edges = pd.read_csv(large_twitch_edges)

    # edges = edges.rename(columns={'numeric_id_1': 'source', 'numeric_id_2': 'target'})

    # print(edges)

    # Creating a simple graph with nodes and edges
    nodes = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'language': ['en', 'fr', 'en', 'de', 'en'],
        'views': [25, 10, 15, 10, 5]
    })

    edges = pd.DataFrame({
        'source': [1, 1, 2, 3, 3],
        'target': [2, 3, 4, 4, 5]
    })


    # Initialize the predictor
    predictor = LinkPredictor(nodes, edges)

    # Create graph and train models
    predictor.create_graph()
    predictor.train_node2vec()
    predictor.train_link_predictor()

    # Predict a link (example: between node 1 and node 2)
    prediction_score = predictor.predict_link(1, 2)
    print(f'Prediction score for link between node 1 and node 2: {prediction_score}')

    # # Predict a link (example: between node 22970 and node 160961)
    # prediction_score = predictor.predict_link(22970, 160961)
    # print(f'Prediction score for link between node 22970 and node 160961: {prediction_score}')


    

    
if __name__ == "__main__":
    configure_logging()
    main()
    