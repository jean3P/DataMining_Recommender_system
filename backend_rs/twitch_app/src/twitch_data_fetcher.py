import requests
import pandas as pd
import logging
from datetime import datetime


class TwitchDataFetcher:
    def __init__(self, username):
        """
        Initialize a fetcher for Twitch user's information.
        :param username: Twitch user's username
        """
        self.username = username
        self.client_id = 'gp762nuuoqcoxypju8c569th9wz7q5'
        self.access_token = 'a9nadaos0k952ndw4qr6fl7i6nj9es'
        self.headers = {
            'Client-ID': self.client_id,
            'Authorization': f'Bearer {self.access_token}'
        }

    def get_twitch_id(self):
        """
        Get only the Twitch user's ID.
        :return: Twitch user's ID as a string or None if not found.
        """
        user_info_url = f'https://api.twitch.tv/helix/users?login={self.username}'
        user_response = self._get_response_json(user_info_url)

        if user_response and 'data' in user_response and len(user_response['data']) > 0:
            user_data = user_response['data'][0]
            return user_data.get('id')
        else:
            logging.error(f"No data found for user: {self.username}")
            return None

    def get_user_info(self):
        """
        Get Twitch user information.
        :return: DataFrame of user information
        """
        user_info_url = f'https://api.twitch.tv/helix/users?login={self.username}'
        stream_info_url = f'https://api.twitch.tv/helix/streams?user_login={self.username}'

        user_info_df = pd.DataFrame(
            columns=['twitch_id', 'created_at', 'affiliated', 'language', 'mature', 'updated_at'])

        user_response = self._get_response_json(user_info_url)
        stream_response = self._get_response_json(stream_info_url)

        if user_response and stream_response:
            user_info_df = self._parse_responses(user_response, stream_response)

        return user_info_df

    def _get_response_json(self, url):
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as err:
            logging.error(f"HTTP error: {err}")
        except requests.exceptions.RequestException as err:
            logging.error(f"Error: {err}")
        return None

    def _parse_responses(self, user_response, stream_response):
        user_data = user_response['data'][0] if user_response['data'] else {}
        stream_data = stream_response['data'][0] if stream_response['data'] else {}

        lang = stream_data.get('language', '').upper()  # Set default to empty string and convert to upper case

        # Handle 'created_at'
        created_at = user_data.get('created_at')
        if created_at:
            create = datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%SZ')
            created_formatted_date = create.strftime('%Y-%m-%d')
        else:
            created_formatted_date = None

        # Handle 'started_at'
        started_at = stream_data.get('started_at')  # Note: This should be from 'stream_data', not 'user_data'
        if started_at:
            update = datetime.strptime(started_at, '%Y-%m-%dT%H:%M:%SZ')
            updated_formatted_date = update.strftime('%Y-%m-%d')
        else:
            updated_formatted_date = None

        user_info = {
            'twitch_id': user_data.get('id'),
            'created_at': created_formatted_date,
            'affiliated': 1 if user_data.get('broadcaster_type') == 'affiliate' else 0,
            'language': lang,
            'mature': int(stream_data.get('is_mature', False)),
            'updated_at': updated_formatted_date
        }
        print(user_info)
        return pd.DataFrame([user_info])

# def main():
#     # # username = 'sanhg5555'
#     # # username = 'porelpepe'
#     # # username = 'Singularidad314'
#     # #username = 'el_pesadito'
#     # username = "Gorgc"
#     # #username = "twitchdev"
#     username = "Gorgc"
#     fetcher = TwitchDataFetcher(username)
#     user_info_df = fetcher.get_user_info()
#     if not user_info_df.empty:
#         print('User Information:\n', user_info_df)
#     else:
#         print("No data retrieved.")
#
#
# configure_logging()
# main()
