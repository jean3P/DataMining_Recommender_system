

from twitch_app.src.classification.ModelTrainer import ModelTrainer
from twitch_app.src.classification.paths import train_80, test_20

forest_trainer = ModelTrainer(
    model_type='svm',
    train_file=train_80,
    test_file=test_20,
    target_col='community_label_leiden',
    drop_col=''
)

# Load, preprocess, and train the model
forest_trainer.load_data()
forest_trainer.preprocess_data()
forest_trainer.train_model()
