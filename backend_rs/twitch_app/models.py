from django.db import models


class TwitchUser(models.Model):
    twitch_id = models.CharField(max_length=100, unique=True)
    created_at = models.CharField(max_length=50)  # Changed to CharField
    affiliated = models.BooleanField(default=False)
    language = models.CharField(max_length=50, default="")
    mature = models.BooleanField(default=False)
    updated_at = models.CharField(max_length=50)  # Changed to CharField

    def __str__(self):
        return self.twitch_id


# class TwitchUserProfile(models.Model):
#     twitch_id = models.CharField(max_length=100, unique=True)
#     created_at = models.DateTimeField()
#     affiliated = models.BooleanField()
#     language = models.CharField(max_length=50, null=True)
#     mature = models.BooleanField()
#     updated_at = models.DateTimeField(auto_now=True)

    # Additional fields as necessary
