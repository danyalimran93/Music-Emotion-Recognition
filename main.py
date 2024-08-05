from FeatureExtractor import FeatureExtractor
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
import pandas as pd

def main():
    train_or_extract = input("Type 'train' to train the model or 'extract' to extract features: ").strip().lower()
    if train_or_extract == 'train':
        main_train()
    elif train_or_extract == 'extract':
        path_to_music_files = input("Path to new music files: ")
        main_extract(path_to_music_files)
    else:
        print("Invalid input!")

# Laster funksjoner fra CSV-filen og trener modellen.
def main_train(): 
    df = pd.read_csv('Emotion_features.csv')
    features = df.iloc[:, 4:].values  # Exclude 'id', 'song_name', 'class', 'label'
    labels = df['label'].values
    
    # Train and evaluate the model
    trainer = ModelTrainer(features, labels)
    trainer.train_and_evaluate()

# Ekstraherer funksjoner fra musikkfiler og oppdaterer databasen.
def main_extract(path_to_music_files): 
    # Initialize FeatureExtractor with the path to the CSV database and the new files directory
    extractor = FeatureExtractor(path_to_music_files) #default database = no 2arg
    extractor.update_database()

if __name__ == "__main__":
    main()