# Imports
import numpy as np
import pandas as pd
import logging
import matplotlib
import math
from collections import defaultdict
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Constants
from constants import (log_file_1, train_80_df, test_20_df,
                       louvain_same_communities_png, leiden_same_communities_png,
                       louvain_not_same_communities_png, leiden_not_same_communities_png, no_community_save_png)

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def configure_logging():
    logging.basicConfig(filename=log_file_1, level=logging.INFO, format="%(message)s", filemode='w')


class DataProcessor:
    @staticmethod
    def load_data():
        """
        Load training and testing datasets from predefined paths.

        Returns:
            tuple: A tuple containing training and testing dataframes.

        Raises:
            FileNotFoundError: If the CSV files are not found at the specified paths.
        """
        try:
            return pd.read_csv(train_80_df), pd.read_csv(test_20_df)
        except FileNotFoundError:
            logging.error("CSV file not found.")
            raise

    @staticmethod
    def preprocess_data(dataset_train, consider_communities=True):
        """
        Preprocess the training dataset by scaling numeric features, passing through binary features,
        and one-hot encoding categorical features.

        Args:
            dataset_train (pd.DataFrame): The training dataset to be preprocessed.
            consider_communities (bool): Whether to consider community labels in preprocessing.

        Returns:
            tuple: A tuple containing the transformed dataset and the preprocessor object.
        """
        # Define the types of features
        numeric_features = ['views']
        binary_features = ['mature']
        categorical_features = ['language']

        # Add community labels to categorical features if consider_communities is True
        if consider_communities:
            categorical_features.extend(['community_label_louvain', 'community_label_leiden'])

        # Create a column transformer for preprocessing
        feature_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),  # Scale numeric features
                ('bin', 'passthrough', binary_features),  # Pass through binary features without changes
                ('cat', OneHotEncoder(), categorical_features)  # One-hot encode categorical features
            ])

        # Return the transformed dataset and the preprocessor
        return feature_preprocessor.fit_transform(dataset_train), feature_preprocessor


class Recommender:
    def __init__(self, train):
        """
        Initialize the Recommender with the training dataset.

        :param train: DataFrame containing the training data.
        """
        self.train = train

    def _get_recommendations(self, row, train_transformed_dataset, feature_preprocessor, same_community=True,
                             consider_communities=False):
        """
        Generate recommendations for a given user based on cosine similarity.

        :param row: DataFrame row representing the user.
        :param train_transformed_dataset: Transformed training dataset.
        :param feature_preprocessor: Preprocessor used for transforming data.
        :param same_community: Boolean indicating if recommendations should be from the same community.
        :param consider_communities: Boolean indicating if communities should be considered in the recommendation logic.
        :return: DataFrame containing recommended users and cosine similarities.
        """
        user_vector = feature_preprocessor.transform(pd.DataFrame([row]))
        cosine_similarities = cosine_similarity(user_vector, train_transformed_dataset)

        if consider_communities:
            # Filter based on community
            louvain_mask = self.train['community_label_louvain'] == row[
                'community_label_louvain'] if same_community else \
                self.train['community_label_louvain'] != row['community_label_louvain']
            leiden_mask = self.train['community_label_leiden'] == row['community_label_leiden'] if same_community else \
                self.train['community_label_leiden'] != row['community_label_leiden']

            # Apply the masks to cosine_similarities
            cosine_similarities[0][~louvain_mask] = -1  # Set to -1 to exclude from argsort
            cosine_similarities[0][~leiden_mask] = -1

        recommended_cosine_indices = cosine_similarities[0].argsort()[-5:][::-1]
        recommended_cosine = self.train.iloc[recommended_cosine_indices]

        return recommended_cosine, cosine_similarities, recommended_cosine_indices

    def generate_recommendations(self, dataset_test, train_transformed_dataset, feature_preprocessor,
                                 same_community=True, consider_communities=True):
        """
        Generate recommendations for a test dataset.

        :param dataset_test: DataFrame containing the test data.
        :param train_transformed_dataset: Transformed training dataset.
        :param feature_preprocessor: Preprocessor used for transforming data.
        :param same_community: Boolean indicating if recommendations should be from the same community.
        :return: Dictionaries containing top cosine scores for Louvain and Leiden communities.
        """
        top_cosine_scores_louvain = defaultdict(list)
        top_cosine_scores_leiden = defaultdict(list)
        top_cosine_scores_no_communities = defaultdict(list)

        if not consider_communities:
            logging.info("\n" + "=" * 80)
            logging.info("RECOMMENDATIONS: NOT COMMUNITIES")
            logging.info("=" * 80 + "\n")

            for _, row in dataset_test.iterrows():
                recommended_cosine, cosine_similarities, recommended_cosine_indices = self._get_recommendations(row,
                                                                                                                train_transformed_dataset,
                                                                                                                feature_preprocessor,
                                                                                                                same_community,
                                                                                                                consider_communities)

                if recommended_cosine.empty:
                    logging.info(
                        f"No recommendations found for user {row['numeric_id']} without considering communities.")
                    continue

                # Sort the recommended_cosine by cosine similarity in descending order
                recommended_cosine = recommended_cosine.copy()
                recommended_cosine['cosine_similarity'] = cosine_similarities[0][recommended_cosine_indices]
                recommended_cosine = recommended_cosine.sort_values(by='cosine_similarity', ascending=False)

                top_cosine_scores_no_communities["No Community"].append(
                    cosine_similarities[0][recommended_cosine_indices[0]])

                user_id = row['numeric_id']
                selected_columns = recommended_cosine[
                    ['numeric_id', 'views', 'mature', 'cosine_similarity']]
                logging.info(f"Recommendations for user {user_id} without considering communities:")
                logging.info(selected_columns.to_string(index=False))
                logging.info("\n" + "=" * 80 + "\n")
            return top_cosine_scores_no_communities, {}

        else:
            if not same_community:
                logging.info("\n" + "=" * 80)
                logging.info("RECOMMENDATIONS FROM DIFFERENT COMMUNITIES")
                logging.info("=" * 80 + "\n")

            for _, row in dataset_test.iterrows():
                recommended_cosine, cosine_similarities, recommended_cosine_indices = self._get_recommendations(row,
                                                                                                                train_transformed_dataset,
                                                                                                                feature_preprocessor,
                                                                                                                same_community,
                                                                                                                consider_communities)

                if recommended_cosine.empty:
                    logging.info(
                        f"No recommendations found for user {row['numeric_id']} {'within the same' if same_community else 'from other'} community.")
                    continue

                # Sort the recommended_cosine by cosine similarity in descending order
                recommended_cosine = recommended_cosine.copy()
                recommended_cosine['cosine_similarity'] = cosine_similarities[0][recommended_cosine_indices]
                recommended_cosine = recommended_cosine.sort_values(by='cosine_similarity', ascending=False)

                louvain_label = recommended_cosine['community_label_louvain'].iloc[0]
                leiden_label = recommended_cosine['community_label_leiden'].iloc[0]
                user_id = row['numeric_id']

                community_types = [('louvain', louvain_label), ('leiden', leiden_label)]

                for community_type, label in community_types:
                    if label:
                        selected_columns = recommended_cosine[
                            ['numeric_id', f'community_label_{community_type}', 'views', 'mature', 'cosine_similarity']]
                        logging.info(f"Recommendations for user {user_id} using:")
                        logging.info(f"Cosine Similarity for {community_type.capitalize()} Community {label}:")
                        logging.info(selected_columns.to_string(index=False))
                        logging.info("\n" + "=" * 80 + "\n")

                top_cosine_scores_louvain[louvain_label].append(cosine_similarities[0][recommended_cosine_indices[0]])
                top_cosine_scores_leiden[leiden_label].append(cosine_similarities[0][recommended_cosine_indices[0]])

            return top_cosine_scores_louvain, top_cosine_scores_leiden


class Statistics:
    @staticmethod
    def compute_statistics(scores_dict):
        stats = {}
        for label, scores in scores_dict.items():
            stats[label] = {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std_dev': np.std(scores),
                'count': len(scores)
            }
        return stats

    @staticmethod
    def compare_statistics(louvain_scores, leiden_scores, community_type="Community"):
        all_labels = set(louvain_scores.keys()).union(set(leiden_scores.keys()))
        sorted_labels = sorted(all_labels, key=lambda x: (isinstance(x, str), x))

        for label in sorted_labels:
            louvain_values = louvain_scores.get(label, [float('nan')])
            leiden_values = leiden_scores.get(label, [float('nan')])

            # Ensure the values are lists
            if not isinstance(louvain_values, list):
                louvain_values = [float('nan')]
            if not isinstance(leiden_values, list):
                leiden_values = [float('nan')]

            louvain_mean = np.mean(louvain_values)
            leiden_mean = np.mean(leiden_values)

            louvain_median = np.median(louvain_values)
            leiden_median = np.median(leiden_values)

            print(f"Comparing {community_type} {label} for Louvain and Leiden:")
            print(
                f"Louvain Mean: {louvain_mean if not np.isnan(louvain_mean) else 'N/A'}, Leiden Mean: {leiden_mean if not np.isnan(leiden_mean) else 'N/A'}")
            print(
                f"Louvain Median: {louvain_median if not np.isnan(louvain_median) else 'N/A'}, Leiden Median: {leiden_median if not np.isnan(leiden_median) else 'N/A'}")
            print("-" * 50)


class Plotter:
    @staticmethod
    def subplot_dimensions(n):
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return rows, cols

    @staticmethod
    def plot_communities(top_cosine_scores, community_type, save_path):
        n_communities = len(top_cosine_scores)
        rows, cols = Plotter.subplot_dimensions(n_communities)

        plt.figure(figsize=(15, 15))

        # Updated sorting logic to handle both string and numeric labels
        sorted_labels = sorted(top_cosine_scores.keys(), key=lambda x: (isinstance(x, str), x))

        for idx, label in enumerate(sorted_labels):
            scores = list(top_cosine_scores[label])
            plt.subplot(rows, cols, idx + 1)
            plt.ylim(0, 1)

            # Adjust color-coding based on community type
            if community_type == "No Community":
                color = 'gray'
            else:
                color = 'blue' if community_type == 'Louvain' else 'green'

            plt.bar(range(len(scores)), scores, color=color, alpha=0.7)

            # Modify title based on community type
            if community_type == "No Community":
                plt.title(f'{community_type}')
            else:
                plt.title(f'{community_type} Community {label}')

            plt.xlabel('User Index')
            plt.ylabel('Similarity Score')
            plt.grid(axis='y')

        plt.tight_layout()

        # Modify suptitle based on community type
        if community_type == "No Community":
            plt.suptitle(f'Top Recommendation Scores: Cosine (Without Communities)', y=1.02)
        else:
            plt.suptitle(f'Top Recommendation Scores: Cosine ({community_type} Communities)', y=1.02)

        plt.savefig(save_path)


def main():
    train, test = DataProcessor.load_data()
    train_transformed, preprocessor_not_communities = DataProcessor.preprocess_data(train, consider_communities=False)

    recommender = Recommender(train)

    # Generate recommendations without considering communities
    top_cosine_scores_no_communities, _ = recommender.generate_recommendations(test, train_transformed,
                                                                               preprocessor_not_communities,
                                                                               consider_communities=False)

    print("=============================================")
    print("=== GENERATING STATS: WITHOUT COMMUNITIES ===")
    print("=============================================")
    Statistics.compare_statistics(top_cosine_scores_no_communities, top_cosine_scores_no_communities, community_type="Option")

    train_transformed, preprocessor = DataProcessor.preprocess_data(train)
    top_cosine_scores_louvain, top_cosine_scores_leiden = recommender.generate_recommendations(test, train_transformed,
                                                                                               preprocessor,
                                                                                               same_community=True)
    print("==========================================")
    print("=== GENERATING STATS: SAME COMMUNITIES ===")
    print("==========================================")
    Statistics.compare_statistics(top_cosine_scores_louvain, top_cosine_scores_leiden)

    top_cosine_scores_louvain_not, top_cosine_scores_leiden_not = recommender.generate_recommendations(test,
                                                                                                       train_transformed,
                                                                                                       preprocessor,
                                                                                                       same_community=False)
    print("===========================================")
    print("=== GENERATING STATS: OTHER COMMUNITIES ===")
    print("===========================================")
    Statistics.compare_statistics(top_cosine_scores_louvain_not, top_cosine_scores_leiden_not)
    print("========================")
    print("=== GENERATING PLOTS ===")
    print("========================")
    Plotter.plot_communities(top_cosine_scores_no_communities, 'No Community', no_community_save_png)
    Plotter.plot_communities(top_cosine_scores_louvain, 'Louvain', louvain_same_communities_png)
    Plotter.plot_communities(top_cosine_scores_leiden, 'Leiden', leiden_same_communities_png)
    Plotter.plot_communities(top_cosine_scores_louvain_not, 'Louvain', louvain_not_same_communities_png)
    Plotter.plot_communities(top_cosine_scores_leiden_not, 'Leiden', leiden_not_same_communities_png)


if __name__ == "__main__":
    configure_logging()
    main()
