# Generated by Django 4.2.7 on 2023-12-02 17:40

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="TwitchUser",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("twitch_id", models.CharField(max_length=100, unique=True)),
                ("created_at", models.DateTimeField()),
                ("affiliated", models.BooleanField(default=False)),
                ("language", models.CharField(max_length=50)),
                ("mature", models.BooleanField(default=False)),
                ("updated_at", models.DateTimeField()),
            ],
        ),
    ]
