# Community-based Content Recommender System

This repository contains a recommendation system that leverages the power of community detection algorithms to generate content suggestions. By analyzing community preferences from Twitch gamer data, the system provides insights and recommendations tailored to individual users.

## Overview

The main workflow of the project can be broken down into three parts:
1. **Community Detection**: Using the `large_twitch_edges.csv` dataset, community detection algorithms (Louvain and Leiden) are applied to group similar users. The results of these algorithms are stored in separate CSV files for each algorithm.
2. **Data Aggregation and Processing**: The main features of users, including their preferences, are loaded from `large_twitch_features.csv`. This data is then merged with the results from the community detection algorithms.
3. **Recommendation Generation**: Recommendations are then generated for users based on either their community preferences alone or a hybrid of their personal and community preferences.

## Important Files

- **constants.py**: Contains the paths to various resource files and directories used throughout the project.
- **large_twitch_edges.csv**: The main data file containing edges (interactions) between Twitch users.
- **large_twitch_features.csv**: Contains features of individual Twitch users including their content preferences.
- **louvain_community_assignments.csv**: Resulting community assignments from the Louvain algorithm.
- **leiden_community_assignments.csv**: Resulting community assignments from the Leiden algorithm.
- **main_communities.py**: The script responsible for applying community detection algorithms on user interactions and generating community assignment files.
- **main_recommender_system.py**: The main script for generating content recommendations.

## Execution Order

1. Run `main_communities.py`. This script will apply community detection algorithms on the `large_twitch_edges.csv` dataset and generate `louvain_community_assignments.csv` and `leiden_community_assignments.csv`.
2. Execute `main_recommender_system.py`. This script will:
   - Load user features and merge them with community assignments.
   - Calculate community preferences based on the `mature` feature.
   - Generate content recommendations for a specified user.
3. Examine the outputs. The merged data will be saved as `large_twitch_features_with_communities.csv` in the `outputs` directory.

## Generated Files

- **louvain_community_assignments.csv**: Contains user nodes and their respective community assignments from the Louvain algorithm.
- **leiden_community_assignments.csv**: Contains user nodes and their respective community assignments from the Leiden algorithm.
- **large_twitch_features_with_communities.csv**: Merged file containing user features along with their community labels from both the Louvain and Leiden algorithms.

## Recommendations

Upon executing `main_recommender_system.py`, content recommendations for a specified user ID will be printed out. These recommendations are based on community preferences, personal preferences, and a hybrid approach combining both.

## Note

Ensure all the paths in the `constants.py` file are correctly set up according to your system's directory structure. Also, make sure all necessary Python libraries are installed and that Python's path settings are appropriately configured for the project's environment.
