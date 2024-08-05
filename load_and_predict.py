import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from FeatureExtractor import FeatureExtractor  # Importer funksjonen fra FeatureExtractor

# Emotion labels and corresponding names
emotion_labels = {1: 'Sad', 2: 'Happy', 3: 'Relax', 4: 'Angry'}
emotion_names = list(emotion_labels.values())

def normalize_features(feature_set, min_max_vals=None):
    for name in feature_set.columns:
        if min_max_vals:
            feature_set[name] = (feature_set[name] - min_max_vals[name][0]) / (min_max_vals[name][1] - min_max_vals[name][0])
        else:
            min_val = feature_set[name].min()
            max_val = feature_set[name].max()
            feature_set[name] = (feature_set[name] - min_val) / (max_val - min_val)
    return feature_set

def load_model(model_path):
    return joblib.load(model_path)

def load_min_max_vals(min_max_file):
    df = pd.read_csv(min_max_file)
    min_max_vals = {row['feature']: (row['min'], row['max']) for _, row in df.iterrows()}
    return min_max_vals

def predict_emotion(model, song_path, min_max_vals):
    # Bruk importert funksjon fra FeatureExtractor
    features_dict = FeatureExtractor.extract_features_from_file(song_path)
    features_df = pd.DataFrame([features_dict])
    features_df = normalize_features(features_df, min_max_vals)
    
    # Feature columns, ensure these match the training data
    feature_columns = [
        'tempo', 'total_beats', 'average_beats', 'chroma_stft_mean', 'chroma_stft_std', 'chroma_stft_var',
        'chroma_cq_mean', 'chroma_cq_std', 'chroma_cq_var', 'chroma_cens_mean', 'chroma_cens_std', 'chroma_cens_var',
        'melspectrogram_mean', 'melspectrogram_std', 'melspectrogram_var', 'mfcc_mean', 'mfcc_std', 'mfcc_var',
        'mfcc_delta_mean', 'mfcc_delta_std', 'mfcc_delta_var','rmse_mean', 'rmse_std', 'rmse_var', 'cent_mean', 
        'cent_std', 'cent_var', 'spec_bw_mean', 'spec_bw_std', 'spec_bw_var', 'contrast_mean', 'contrast_std', 
        'contrast_var', 'rolloff_mean', 'rolloff_std', 'rolloff_var', 'poly_mean', 'poly_std', 'poly_var', 'tonnetz_mean',
        'tonnetz_std', 'tonnetz_var', 'zcr_mean', 'zcr_std', 'zcr_var', 'harm_mean', 'harm_std', 'harm_var', 'perc_mean',
        'perc_std', 'perc_var', 'frame_mean', 'frame_std', 'frame_var'
    ]

    # Ensure the extracted features include all required columns
    missing_cols = set(feature_columns) - set(features_df.columns)
    if missing_cols:
        raise ValueError(f"Extracted features do not match the training dataset feature columns. Missing columns: {missing_cols}")

    features_df = features_df[feature_columns]
    
    # Get prediction probabilities
    prediction_probs = model.predict_proba(features_df)[0]
    predicted_label = np.argmax(prediction_probs) + 1
    predicted_emotion = emotion_labels[predicted_label]
    confidence = prediction_probs[predicted_label - 1] * 100
    
    return predicted_emotion, confidence, features_df, prediction_probs

def plot_emotion_prediction(confidence, prediction_probs):
    # Visualize prediction confidence levels
    plt.figure(figsize=(10, 6))
    
    sns.barplot(x=emotion_names, y=prediction_probs * 100)
    plt.title(f'Emotion Prediction Confidence (Confidence: {confidence:.2f}%)')
    plt.ylabel('Confidence (%)')
    plt.xlabel('Emotions')
    plt.ylim(0, 100)
    plt.show()

if __name__ == "__main__":
    model_path = 'Emotion_Audio_Detection_Model.pkl'
    song_path = input("path to song that will be used for prediction: ")
    min_max_file = 'min_max_vals.csv'  # Optional if you save Min/Max values for normalization
    
    try:
        model = load_model(model_path)
        if os.path.exists(min_max_file):
            min_max_vals = load_min_max_vals(min_max_file)
        else:
            min_max_vals = None
            
        predicted_emotion, confidence, features_df, prediction_probs = predict_emotion(model, song_path, min_max_vals)
        print(f"Predicted Emotion: {predicted_emotion}, Confidence: {confidence:.2f}%")
        for emotion, prob in zip(emotion_names, prediction_probs):
            print(f"{emotion}: {prob * 100:.2f}%")
        
        plot_emotion_prediction(confidence, prediction_probs)
    except FileNotFoundError as e:
        print("Error loading model or files:", e)
    except ValueError as e:
        print("Error in feature extraction or prediction:", e)
