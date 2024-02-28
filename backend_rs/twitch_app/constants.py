import os

import sys

abs_path = sys.path[0]

base_name = os.path.dirname(abs_path)
model_path = os.path.join("twitch_app/src/classification/svm_model.joblib")
community_labels = [0, 1, 2, 3, 6, 13, 36, 38, 44, 53, 59, 80, 86, 91, 93, 95, 108, 114]

