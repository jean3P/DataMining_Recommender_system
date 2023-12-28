import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import logging

import time

from constants import log_file_2, large_twitch_edges, large_twitch_features, outputs_path, louvain_file, leiden_file

def configure_logging():
    logging.basicConfig(filename=log_file_2, level=logging.INFO, format="%(message)s", filemode='w')


# Set seed for reproducibility
np.random.seed(0)


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
        self.lr_model = None
        self.auc_score = None



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
            # node2vec = Node2Vec(self.G, dimensions=64, walk_length=30, num_walks=200, workers=4)
            node2vec = Node2Vec(self.G, dimensions=10, walk_length=10, num_walks=50, workers=4)


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
                examples[['source', 'target']], examples['label'], test_size=0.3, random_state=0)

            # Apply prediction to the dataset
            X_train['similarity'] = [self.predict_link(row['source'], row['target']) for index, row in X_train.iterrows()]

            # Train a logistic regression model
            self.lr_model = LogisticRegression(random_state=0)
            self.lr_model.fit(X_train[['similarity']], y_train)

            # Predict on the test set
            X_test['similarity'] = [self.predict_link(row['source'], row['target']) for index, row in X_test.iterrows()]
            y_pred = self.lr_model.predict_proba(X_test[['similarity']])[:, 1]

            # Evaluate the model
            self.auc_score = roc_auc_score(y_test, y_pred)
            print(f'The AUC score of the link prediction model is: {self.auc_score}')
        
        except self.G is None:
            logging.error("Need to create a graph.")

        except self.model is None:
            logging.error("Need to train model.")


    def get_auc_score(self):
        try:
            return self.auc_score
        
        except self.G is None:
            logging.error("Need to create a graph.")

        except self.model is None:
            logging.error("Need to train model.")
        
        except self.auc_score is None:
            logging.error("Need to train model.")
    
    def predict_link(self, source, target):
        try:
            vector_1 = self.model.wv.get_vector(str(source))
            vector_2 = self.model.wv.get_vector(str(target))

            return np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
        
        except self.model is None:
            logging.error("Need to train model.")


    def predict_probability(self, source, target):
        """
        Predict the probability of a link between two nodes using the trained logistic regression model.
        
        :param source: ID of the first node
        :param target: ID of the second node
        :return: Probability of a link as predicted by the logistic regression model
        """
        # Ensure the node2vec model has been trained
        if self.model is None:
            raise ValueError("The node2vec model needs to be trained before predicting.")

        # Ensure the logistic regression model has been trained
        if self.lr_model is None:
            raise ValueError("The logistic regression model needs to be trained before predicting.")

        # Calculate the similarity score using the node2vec embeddings
        similarity = self.predict_link(source, target)

        # Create a DataFrame with the similarity score, with column name that matches the training data
        similarity_score_reshaped = pd.DataFrame([[similarity]], columns=['similarity'])

        # Predict the probability using the logistic regression model
        probability = self.lr_model.predict_proba(similarity_score_reshaped)[:, 1][0]
        return probability


    
def main():

    # Example of the link predictor usage

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


    # Starting time counting
    start_time = time.time()

    # Initialize the predictor
    predictor = LinkPredictor(nodes, edges)

    # Create graph and train models
    predictor.create_graph()
    predictor.train_node2vec()
    predictor.train_link_predictor()

    # Predict a link (example: between node 1 and node 2)
    prediction_score = predictor.predict_link(2, 3)
    print(f'Similarity score for link between node 2 and node 3: {prediction_score}')

    print("Process finished --- %s seconds ---" % (time.time() - start_time))

    # Predict the probability of a link between node 1 and node 2
    link_probability = predictor.predict_probability(2, 3)
    print(f'Probability of link between node 2 and node 3: {link_probability}')
    

    
if __name__ == "__main__":
    configure_logging()
    main()
    