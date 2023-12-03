# twitch_app/management/commands/clear_twitchusers.py

from django.core.management.base import BaseCommand
from twitch_app.models import TwitchUser

class Command(BaseCommand):
    help = 'Clears the TwitchUser table and resets IDs'

    def handle(self, *args, **options):
        # Clear the TwitchUser table
        TwitchUser.objects.all().delete()

        self.stdout.write(self.style.SUCCESS('Successfully cleared TwitchUser table'))

        # Note: Resetting IDs is database dependent; below is an example for PostgreSQL
        # from django.db import connection
        # with connection.cursor() as cursor:
        #     cursor.execute("ALTER SEQUENCE twitch_app_twitchuser_id_seq RESTART WITH 1")
