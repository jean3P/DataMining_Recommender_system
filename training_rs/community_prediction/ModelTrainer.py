import os

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

from constants import path_root_backend_classification_directory


def plot_confusion_matrix(y_true, y_pred, model_type):
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Set community numbers
    communities = [0, 1, 2, 3, 6, 13, 36, 38, 44, 53, 59, 80, 86, 91, 93, 95, 108, 114]

    # Create a heatmap from the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=communities, yticklabels=communities) # Here we set custom labels
    plt.xlabel('Predicted Community')
    plt.ylabel('True Community')
    plt.title(f'Confusion Matrix - {model_type}')
    plt.show()


class ModelTrainer:
    def __init__(self, model_type, train_file, test_file, target_col, drop_col=None):
        self.preprocessor = None
        self.test_data = None
        self.train_data = None
        self.model_type = model_type
        self.train_file = train_file
        self.test_file = test_file
        self.target_col = target_col
        self.drop_col = drop_col
        self.pipeline = None

    def load_data(self):
        self.train_data = pd.read_csv(self.train_file)
        self.test_data = pd.read_csv(self.test_file)

        if self.drop_col:
            self.train_data.drop(self.drop_col, axis=1, inplace=True)
            self.test_data.drop(self.drop_col, axis=1, inplace=True)

    def preprocess_data(self):
        for df in [self.train_data, self.test_data]:
            # Handle Date Features
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['updated_at'] = pd.to_datetime(df['updated_at'])
            df['account_age'] = (df['updated_at'] - df['created_at']).dt.days
            df.drop(['created_at', 'updated_at'], axis=1, inplace=True)

            # Identifying columns for transformation
        categorical_cols = ['language']  # Add other categorical columns if needed
        binary_cols = ['mature', 'affiliate']  # Assuming binary format

        # Define transformations
        transformers = [
            ('cat', OneHotEncoder(), categorical_cols)
        ]

        # Initialize ColumnTransformer
        self.preprocessor = ColumnTransformer(transformers=transformers)

    def perform_cross_validation(self, n_splits=5):
        # Define feature columns
        feature_cols = self.train_data.columns.drop(self.target_col)

        # Separate features and target variable in training data
        X = self.train_data[feature_cols]
        y = self.train_data[self.target_col]

        # Initialize k-Fold cross-validation
        kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)

        # Initialize and train the model based on model_type
        if self.model_type == 'logistic_regression':
            model = LogisticRegression(solver='saga', max_iter=80000)
        elif self.model_type == 'decision_tree':
            model = DecisionTreeClassifier(random_state=42)
        elif self.model_type == 'svm':
            model = SVC()
        else:
            raise ValueError("Invalid model type. Choose 'logistic_regression' or 'decision_tree'.")

        # Perform cross-validation
        cv_scores = cross_val_score(Pipeline([('preprocessor', self.preprocessor), ('classifier', model)]), X, y, cv=kf)

        # Print cross-validation results
        print(f"CV Scores: {cv_scores}")
        print(f"Average CV Score: {cv_scores.mean()}")
        return cv_scores.mean()

    def train_model(self):
        # if self.perform_cross_validation() < 0.5 :
        #     print(f"The mean score is less that 50%")
        # else:
        # Define feature columns
        feature_cols = self.train_data.columns.drop(self.target_col)

        # Separate features and target variable in training and test data
        X_train = self.train_data[feature_cols]
        y_train = self.train_data[self.target_col]
        X_test = self.test_data[feature_cols]
        y_test = self.test_data[self.target_col]

        # Initialize and train the model based on model_type
        if self.model_type == 'logistic_regression':
            self.pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', LogisticRegression(solver='saga', max_iter=80000))
            ])
        elif self.model_type == 'decision_tree':
            self.pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', DecisionTreeClassifier(random_state=42))
            ])
        elif self.model_type == 'random_forest':
            self.pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', RandomForestClassifier(random_state=42))
            ])
        elif self.model_type == 'svm':
            self.pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', SVC(probability=True))
            ])
        else:
            raise ValueError(
                "Invalid model type. Choose 'logistic_regression', 'decision_tree', 'random_forest', or 'svm'.")

        # Train the model
        self.pipeline.fit(X_train, y_train)
        path_model = os.path.join(path_root_backend_classification_directory, 'svm_model.joblib')
        dump(self.pipeline, path_model)
        # Evaluate the model
        y_pred = self.pipeline.predict(X_test)
        print(classification_report(y_test, y_pred, zero_division=0))
        plot_confusion_matrix(y_test, y_pred, self.model_type)
