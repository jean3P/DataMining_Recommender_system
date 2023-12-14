import requests
import pandas as pd
import json

import logging

from constants import log_file_3, large_twitch_edges, large_twitch_features, outputs_path, louvain_file, leiden_file

def configure_logging():
    logging.basicConfig(filename=log_file_3, level=logging.INFO, format="%(message)s", filemode='w')


class TwitchDataFetcher:
    def __init__(self, username):
        """
        Initialize a fetcher for new user's information using Twtich API.

        :param username: New user Twitch name
        """

        self.username = username
        self.client_id = 'o20wg0edq81zrbi4k44adg7qybf4vd'
        self.access_token = 'm7j4z202jqbcz652y3n0k66er4ftv5'

        self.headers = {'Client-ID': self.client_id,
                        'Authorization': f'Bearer {self.access_token}'
                        }

    def get_user_info(self):
        """
        Get user information

        :return: Dataframe of user information
        """

        # self.username = "twitchdev"

        user_features_df = pd.DataFrame()
        user_features_df['Id'] = None
        # user_features_df['views'] = [int(view_count)]
        user_features_df['mature'] = None
        # user_features_df['life_time'] = None
        user_features_df['created_at'] = None
        user_features_df['updated_at'] = None
        # user_features_df['dead_account'] = None
        user_features_df['language'] = None
        user_features_df['affiliated'] = None

        features = pd.read_csv(large_twitch_features)
        features = features.rename(columns={'numeric_id': 'Node'})



        response_user = requests.get(f'https://api.twitch.tv/helix/users?login={self.username}', headers=self.headers)

        response_stream_last = requests.get(f'https://api.twitch.tv/helix/streams?user_login={self.username}', headers=self.headers)

        response_user_json = response_user.json()  # Convert the response to JSON format
        response_stream_last_json = response_stream_last.json()

        # print(response_user_json)

        # Check if the data key exists and has at least one item
        if 'data' in response_user_json and len(response_user_json['data']) > 0:

            user_data = response_user_json['data'][0]  # Extract the first item from the data list

            # Extract individual fields
            user_id = user_data.get('id', None)

            if user_id in features['Node'].values:
                user_features_df['Id'] = [int(user_id)]

                user_features_df['mature'] = 1
                user_features_df['created_at'] = 1
                user_features_df['updated_at'] = 1
                # user_features_df['dead_account'] = 1
                user_features_df['language'] = 1
                user_features_df['affiliated'] = 1

            else:
                username = user_data.get('login', None)
                broadcaster_type = user_data.get('broadcaster_type', None)
                # view_count = user_data.get('view_count', 0) # This field has got deprecated
                created_at = user_data.get('created_at', None)

                # Saving in dataframe
                user_features_df['Id'] = [int(user_id)]
                # user_features_df['views'] = [int(view_count)]
                user_features_df['created_at'] = [created_at]
                if broadcaster_type == 'affiliated':
                    user_features_df['affiliated'] = [1]
                else:
                    user_features_df['affiliated'] = [0]

        else:
            print("No user data found.")

        if 'data' in response_stream_last_json and len(response_stream_last_json['data']) > 0:

            last_stream_data = response_stream_last_json['data'][0]  # Extract the first item from the data list

            # Extract individual fields
            last_stream_language = last_stream_data.get('language', None)
            last_stream_mature = last_stream_data.get('is_mature', None)
            last_stream_started_at = last_stream_data.get('started_at', None)

            # Saving in dataframe
            user_features_df['language'] = [last_stream_language]
            user_features_df['mature'] = [int(last_stream_mature)]
            user_features_df['updated_at'] = [last_stream_started_at]
        else:
            print("No last stream data found.")


        
        return user_features_df
    
def get_username(user_id):
    
    username = None

    client_id = 'o20wg0edq81zrbi4k44adg7qybf4vd'
    access_token = 'm7j4z202jqbcz652y3n0k66er4ftv5'

    headers = {'Client-ID': client_id,
                    'Authorization': f'Bearer {access_token}'
                    }
    
    response = requests.get(f'https://api.twitch.tv/helix/users?id={user_id}', headers=headers)

    if response.status_code == 200:
        if len(response.json()['data'])>0:
            user_data = response.json()['data'][0]
            username = user_data['login']  # The 'login' field contains the username
            print(f"Username: {username}")
        else:
            print("There is no user data available")
    else:
        print(f"Failed to retrieve user data: {response.status_code}")

    return username




        # response_stream_first =  requests.get(f'https://api.twitch.tv/helix/videos?user_id={user_id}&sort=time&first=1', headers=self.headers)

        # response_stream_first_json = response_stream_first.json()

        # print(response_stream_first_json)

        # if 'data' in response_stream_first_json and len(response_stream_first_json['data']) > 0:

        #     first_stream_data = response_stream_first_json['data'][0]  # Extract the first item from the data list

        #     # Extract individual fields
        #     last_stream_language = last_stream_data.get('language', 'Not available')
        #     last_stream_mature = last_stream_data.get('is_mature', 'Not available')
        #     last_stream_started_at = last_stream_data.get('started_at', 'Not available')

        #     # Saving in dataframe
        #     user_features_df['language'] = last_stream_language
        #     user_features_df['mature'] = last_stream_mature
        #     user_features_df['updated_at'] = last_stream_started_at
        # else:
        #     print("No first stream data found.")
    



def main():
    # username = 'sanhg5555'
    # username = 'porelpepe'
    # username = 'Singularidad314'
    #username = 'el_pesadito'
    # username = "Gorgc"
    username = "twitchdev"
    username = "944u3gva6ksunn6t"

    # Initializing the fetcher with the username
    fetcher = TwitchDataFetcher(username)


    # Fetching the user features
    features_df = fetcher.get_user_info()
    print(features_df.dtypes)
    print('Features of the new user: \n ', features_df)

    new_user_id = 495 # username="aphasia"
    new_user_id = 8216 # username = "944u3gva6ksunn6t"
    new_user_id = 62379 # username = "carlosm"
    new_user_id = 10660 # username = "mamasitabatwoman"
    new_user_id = 7574 # username = "qbexo"
    new_user_id = 14241 # username = "kong"
    new_user_id = 78 # username = "mashuiradu"
    new_user_id = 2321 # username = "q2gbsksdsqcevlg"

    print(f'Username of the user: {new_user_id} is: {get_username(new_user_id)}')



    
if __name__ == "__main__":
    
    configure_logging()
    main()
    