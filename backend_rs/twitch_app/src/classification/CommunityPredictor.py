import json
from joblib import load
import numpy as np


class CommunityPredictor:
    """

    CommunityPredictor class represents a predictor for community labels based on a trained model.

    Attributes:
    - model (object): The trained model used for prediction.
    - community_labels (list): The labels corresponding to different communities.
    - prediction_df (object): The data used for prediction.

    Methods:
    - __init__(self, model_path, community_labels, prediction_df):
        Initializes a CommunityPredictor object.
        Args:
            model_path (str): The path to the serialized model.
            community_labels (list): The labels corresponding to different communities.
            prediction_df (object): The data used for prediction.

    - predict(self):
        Generates predictions for community labels.
        Returns:
            str: JSON-string representation of predictions and probabilities.

    """
    def __init__(self, model_path, community_labels, prediction_df):
        self.model = load(model_path)
        self.community_labels = community_labels
        self.prediction_df = prediction_df

    def predict(self):
        # Generate predictions and probabilities
        print(self.prediction_df)
        probabilities = self.model.predict_proba(self.prediction_df)
        # predictions = self.model.predict(self.prediction_df)

        # Find the index of the highest probability
        highest_prob_index = np.argmax(probabilities, axis=1)

        # Map the index to the actual community label
        predicted_communities = [self.community_labels[index] for index in highest_prob_index]

        # Get the highest probability value for each prediction
        highest_probs = probabilities[np.arange(len(probabilities)), highest_prob_index]*100

        # Construct the result JSON
        result = [{"community": community, "probability": f"{prob:.2f}%" } for community, prob in
                  zip(predicted_communities, highest_probs)]
        return json.dumps(result, indent=4)

# Usage example
# community_labels = [0, 1, 2, 3, 8, 16, 28, 38, 41, 42, 50, 57, 72, 81, 85, 89, 111]
# model_path = "svm_model.joblib"
# prediction_df = processor.process_data()  # Assuming 'processor' is an instance of your data processing class
#
# predictor = CommunityPredictor(model_path, community_labels, prediction_df)
# result_json = predictor.predict()
# print(result_json)
