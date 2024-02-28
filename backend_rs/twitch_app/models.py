from django.db import models


class TwitchUser(models.Model):
    """

    TwitchUser Model Class

    Attributes:
    - twitch_id (str): The unique identifier of the Twitch user.
    - created_at (str): The timestamp of when the Twitch user's account was created.
    - affiliated (bool): Indicates if the Twitch user is affiliated (True) or not (False).
    - language (str): The language preference of the Twitch user.
    - mature (bool): Indicates if the Twitch user's content is for mature audiences (True) or not (False).
    - updated_at (str): The timestamp of when the Twitch user's account was last updated.

    Methods:
    - __str__(): Returns a string representation of the TwitchUser object, containing the twitch_id.

    Usage Example:
        user = TwitchUser.objects.create(twitch_id="mytwitchid", created_at="2022-01-01", affiliated=True, language="en", mature=False, updated_at="2022-01-02")
        print(user)

    Output:
        mytwitchid

    """
    twitch_id = models.CharField(max_length=100, unique=True)
    created_at = models.CharField(max_length=50)  # Changed to CharField
    affiliated = models.BooleanField(default=False)
    language = models.CharField(max_length=50, default="")
    mature = models.BooleanField(default=False)
    updated_at = models.CharField(max_length=50)  # Changed to CharField

    def __str__(self):
        return self.twitch_id
