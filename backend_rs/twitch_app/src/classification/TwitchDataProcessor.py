import pandas as pd
from datetime import datetime

from joblib import load


class TwitchDataProcessor:
    def __init__(self, user_data):
        self.user_data = user_data

    def process_data(self):
        # Convert dates to datetime objects
        if 'updated_at' not in self.user_data or self.user_data['updated_at'] is None:
            updated_at = datetime.strptime(self.user_data['created_at'], '%Y-%m-%d')
        else:
            updated_at = datetime.strptime(self.user_data['updated_at'], '%Y-%m-%d')

        created_at = datetime.strptime(self.user_data['created_at'], '%Y-%m-%d')

        # Calculate account_age (in days)
        account_age = (updated_at - created_at).days

        # Assuming some dummy values for missing columns
        views = 50000  # Example value
        numeric_id = 123456  # Example value
        dead_account = 0  # Example value

        # Prepare the DataFrame for prediction
        data_for_prediction = {
            'views': views,
            'mature': int(self.user_data['mature']),
            'life_time': account_age,
            'created_at': created_at,
            'updated_at': updated_at,
            'numeric_id': numeric_id,
            'dead_account': dead_account,
            'language': self.user_data['language'],
            'affiliate': int(self.user_data['affiliated'])
        }

        # Converting to DataFrame
        prediction_df = pd.DataFrame([data_for_prediction])

        return prediction_df


# # Usage Example
# user_data = {'twitch_id': '206981151', 'created_at': '2018-03-20',
#              'affiliated': 0, 'language': 'ES', 'mature': 0,
#              'updated_at': '2023-12-01'}
#
# processor = TwitchDataProcessor(user_data)
# prediction_df = processor.process_data()
# print(prediction_df.to_string(index=False))
#
# svm_model = load("svm_model.joblib")
#
# # Assuming svm_model is an instance of SVC with probability=True
# probabilities = svm_model.predict_proba(prediction_df)
# print("Probabilities:\n", probabilities)
#
# # decision_function = svm_model.decision_function(prediction_df)
# # print("Decision Function:\n", decision_function)
#
# predictions = svm_model.predict(prediction_df)
# print("Community predicted:")
# print(predictions)
#
# # Community labels
# community_labels = [0, 1, 2, 3, 8, 16, 28, 38, 41, 42, 50, 57, 72, 81, 85, 89, 111]
#
# # Model's prediction (index)
# predicted_index = 2  # This is the output from your model
#
# # Map the index to the actual community label
# predicted_community = community_labels[predicted_index]
#
# print("Predicted Community Label:", predicted_community)