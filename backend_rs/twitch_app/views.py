from django.http import JsonResponse
from .models import TwitchUser
from twitch_app.src.twitch_data_fetcher import TwitchDataFetcher  # Ensure this is the correct import for your fetcher class


def fetch_twitch_data(request, username):
    fetcher = TwitchDataFetcher(username)
    user_info_df = fetcher.get_user_info()
    print(user_info_df)
    if not user_info_df.empty:
        user_info = user_info_df.iloc[0].to_dict()
        print(user_info)
        twitch_user, created = TwitchUser.objects.update_or_create(
            twitch_id=user_info['twitch_id'],  # Use 'twitch_id' here
            defaults=user_info
        )
        return JsonResponse(user_info)

    return JsonResponse({'error': 'No data retrieved.'})

