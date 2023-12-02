import requests
import pandas as pd
import logging


class TwitchDataFetcher:
    def __init__(self, username):
        """
        Initialize a fetcher for Twitch user's information.
        :param username: Twitch user's username
        """
        self.username = username
        self.client_id = 'o20wg0edq81zrbi4k44adg7qybf4vd'
        self.access_token = 'm7j4z202jqbcz652y3n0k66er4ftv5'
        self.headers = {
            'Client-ID': self.client_id,
            'Authorization': f'Bearer {self.access_token}'
        }

    def get_user_info(self):
        """
        Get Twitch user information.
        :return: DataFrame of user information
        """
        user_info_url = f'https://api.twitch.tv/helix/users?login={self.username}'
        stream_info_url = f'https://api.twitch.tv/helix/streams?user_login={self.username}'

        user_info_df = pd.DataFrame(columns=['twitch_id', 'created_at', 'affiliated', 'language', 'mature', 'updated_at'])

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

        user_info = {
            'twitch_id': user_data.get('id'),
            'created_at': user_data.get('created_at'),
            'affiliated': 1 if user_data.get('broadcaster_type') == 'affiliate' else 0,
            'language': stream_data.get('language'),
            'mature': int(stream_data.get('is_mature', False)),
            'updated_at': stream_data.get('started_at')
        }

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
