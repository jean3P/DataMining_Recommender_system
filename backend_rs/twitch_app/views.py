import json
from datetime import datetime
from django.http import JsonResponse
import os
from .constants import community_labels, model_path
from .models import TwitchUser
from twitch_app.src.twitch_data_fetcher import \
    TwitchDataFetcher  # Ensure this is the correct import for your fetcher class
from .src.classification.CommunityPredictor import CommunityPredictor
from .src.classification.TwitchDataProcessor import TwitchDataProcessor
from .src.classification.TwitchRecommender import TwitchRecommender


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
    fetcher = TwitchDataFetcher(username)
    twitch_id = fetcher.get_twitch_id()

    if twitch_id:
        # Check if user already exists in database
        try:
            twitch_user = TwitchUser.objects.get(twitch_id=twitch_id)
            user_info = format_twitch_user(twitch_user)
        except TwitchUser.DoesNotExist:
            # If user does not exist, fetch user info and process
            user_info_df = fetcher.get_user_info()
            if not user_info_df.empty:
                user_info = user_info_df.iloc[0].to_dict()

        # Process the data for prediction
        processor = TwitchDataProcessor(user_info)
        prediction_df = processor.process_data()
        print(prediction_df)
        m_p = os.path.join(os.getcwd(), model_path)
        predictor = CommunityPredictor(m_p, community_labels, prediction_df)
        community_prediction = predictor.predict()

        # Parse the JSON result
        community_prediction = json.loads(community_prediction)[0]  # Assuming single prediction

        # Fetch Recommendations
        print(twitch_id)
        print(community_prediction['community'])
        recommendations_json = TwitchRecommender(twitch_id, community_prediction['community'])
        recommendations = json.loads(recommendations_json)

        #
        # # Save new user info to database
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
