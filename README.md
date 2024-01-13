# DataMining Recommender System for Twitch (TWITCHCOMM)

## Overview
The DataMining Recommender System is designed to enhance the user experience on Twitch by providing personalized streamer recommendations. This system leverages data mining techniques to predict communities within the Twitch social network and recommend streamers based on these community affiliations.

## Repository Structure
- `backend_rs`: Contains the Django backend server, responsible for data processing, API management, and serving recommendation data.
  - `twitch_app`: Core application handling Twitch data processing and community prediction.
- `frontend_rs`: React-based frontend integrated with Electron, offering a user-friendly interface for displaying recommendations.
- `training_rs`: Includes scripts and resources for training the machine learning models used in community prediction.
  - `community_prediction`: Directory dedicated to developing models for predicting Twitch communities.
- `LICENSE`: Project's license file.
- `README.md`: This file, providing an overview and instructions for the project.

## Setup and Installation
1. **Backend Setup:**
   - Navigate to the `backend_rs` directory.
   - Install required dependencies: `pip install -r requirements.txt`
   - Run the Django server: `python manage.py runserver`

2. **Frontend Setup:**
   - Go to the `frontend_rs` directory.
   - Install necessary packages: `npm install`
   - Start the React application: `npm start`

3. **Training the Models:**
   - In the `training_rs` directory, run the training scripts to generate the community prediction models.

## Usage
- After starting the Django server and the React application, access the frontend via a web browser.
- Enter your Twitch username to receive personalized streamer recommendations.

## Contributing
Contributions to the DataMining Recommender System are welcome. Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments
- Twitch API
- Community contributors and users of the system
