import json
from datetime import datetime

import pandas as pd
from django.http import JsonResponse
import os
from .constants import community_labels, model_path
from twitch_app.src.twitch_data_fetcher import \
    TwitchDataFetcher
from .models import TwitchUser
from .src.classification.CommunityPredictor import CommunityPredictor
from .src.classification.TwitchDataProcessor import TwitchDataProcessor
from .src.classification.TwitchRecommender import TwitchRecommenderSystem
from .src.classification.paths import large_twitch_features


def format_twitch_user(twitch_user):
    """
    Converts TwitchUser object fields into a dictionary,
    with date strings converted to the desired format.
    """

    def format_date(date_str):
        if date_str:
            return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        return None

    return {
        'twitch_id': twitch_user.twitch_id,
        'created_at': format_date(twitch_user.created_at),
        'affiliated': twitch_user.affiliated,
        'language': twitch_user.language,
        'mature': twitch_user.mature,
        'updated_at': format_date(twitch_user.updated_at),
        # Add other fields as needed
    }


def save_date_in_format(date_str, format_to_save='%Y-%m-%d %H:%M:%S'):
    """Converts a date string to a specified format for saving."""
    if date_str:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime(format_to_save)
    return None


def fetch_twitch_data(request, username):
    """
    Fetches Twitch data for a given username.

    :param request: The request object.
    :param username: The username of the Twitch user.
    :return: A JsonResponse containing the fetched Twitch data or an error message.

    Example usage:
    fetch_twitch_data(request, 'example_username')
    """
    fetcher = TwitchDataFetcher(username)
    twitch_id = fetcher.get_twitch_id()

    features = pd.read_csv(large_twitch_features)
    print(fetcher)
    user_id = int(twitch_id)
    # Filter the DataFrame to find the row with the matching numeric_id
    matched_row = features[features['numeric_id'] == user_id]
    flag = False
    if not matched_row.empty:
        # Get the first (and should be only) row of the matched data
        user_data = matched_row.iloc[0]
        flag = True
        user_pretrained = {
            'twitch_id': twitch_id,
            'created_at': user_data['created_at'],
            'affiliated': bool(user_data['affiliate']),
            'language': user_data['language'],
            'mature': bool(user_data['mature']),
            'updated_at': user_data['updated_at']
        }

        user_pretrained = pd.DataFrame([user_pretrained])

        print(user_pretrained)
    else:
        print(f"No data found for user ID {user_id}")

    if twitch_id:
        # Check if user already exists in database
        try:
            twitch_user = TwitchUser.objects.get(twitch_id=twitch_id)
            user_info = format_twitch_user(twitch_user)
        except TwitchUser.DoesNotExist:
            # If user does not exist, fetch user info and process
            if flag:
                user_info_df = user_pretrained
            else:
                user_info_df = fetcher.get_user_info()
            if not user_info_df.empty:
                user_info = user_info_df.iloc[0].to_dict()

        if 'updated_at' not in user_info or user_info['updated_at'] is None:
            user_info.update({'updated_at': user_info.get('created_at')})
        if user_info.get('language') == '':
            user_info.update({'language': 'EN'})
        processor = TwitchDataProcessor(user_info)
        prediction_df = processor.process_data()

        m_p = os.path.join(os.getcwd(), model_path)
        predictor = CommunityPredictor(m_p, community_labels, prediction_df)
        community_prediction = predictor.predict()

        # Parse the JSON result
        community_prediction = json.loads(community_prediction)[0]  # Assuming single prediction

        # Fetch Recommendations
        print(twitch_id)
        print(community_prediction['community'])
        recommender_system = TwitchRecommenderSystem()
        recommendations_json = recommender_system.twitch_recommender(twitch_id, community_prediction['community'])
        recommendations = json.loads(recommendations_json)

        # Save new user info to database
        # Convert dates to the desired format before saving
        user_info['created_at'] = save_date_in_format(user_info['created_at'])
        user_info['updated_at'] = save_date_in_format(user_info['updated_at'])

        # Update or create the TwitchUser instance
        twitch_user, created = TwitchUser.objects.update_or_create(
            twitch_id=user_info['twitch_id'],
            defaults=user_info
        )

        # Add community prediction to the response
        user_info.update(community_prediction)
        user_info.update(recommendations)

        return JsonResponse(user_info)

    return JsonResponse({'error': 'Username not found.'})

# LaMediaInglesa
# paaulacg_
# LoLWorldChampionship
