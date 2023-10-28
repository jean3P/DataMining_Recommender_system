import os

import pandas as pd
from sklearn.model_selection import train_test_split
from constants import train_df, valid_df, test_df, features_with_communities, train_80_df, test_20_df


def split_data(df, train_size=0.7, validation_size=0.15, test_size=0.15):
    train, temp = train_test_split(df, test_size=1 - train_size, random_state=42)
    valid, test = train_test_split(temp, test_size=test_size / (test_size + validation_size), random_state=42)
    return train, valid, test


def hold_out_split(df, hold_out_size=0.2):
    # Split the data into 80% for generating recommendations and 20% for testing
    train, test = train_test_split(df, test_size=hold_out_size, random_state=42)
    return train, test


features_df = pd.read_csv(features_with_communities)

if not (os.path.exists(train_df) and os.path.exists(valid_df) and os.path.exists(test_df)
        and os.path.exists(train_80_df) and os.path.exists(test_20_df)):
    train_data, valid_data, test_data = split_data(features_df)

    train_data.to_csv(train_df, index=False)
    valid_data.to_csv(valid_df, index=False)
    test_data.to_csv(test_df, index=False)

    train_data_80, test_data_20 = hold_out_split(features_df)
    train_data_80.to_csv(train_80_df, index=False)
    test_data_20.to_csv(test_20_df, index=False)

else:
    print("Divided datasets already exist!")
