# Generated by Django 4.2.7 on 2023-12-02 20:39

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("twitch_app", "0002_twitchuserprofile"),
    ]

    operations = [
        migrations.AlterField(
            model_name="twitchuserprofile",
            name="language",
            field=models.CharField(max_length=50, null=True),
        ),
    ]