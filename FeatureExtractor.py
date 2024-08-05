import os
import pandas as pd
import numpy as np
import joblib
import librosa

class FeatureExtractor:
    def __init__(self, path, database_csv='Emotion_features.csv', model_path='Emotion_Audio_Detection_Model.pkl'):
        self.path = path
        self.database_csv = database_csv
        self.emotion_map = {
            'happy': 2,
            'sad': 1,
            'angry': 4,
            'relaxed': 3
        }
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = None

    @staticmethod
    def extract_features_from_file(songname):
        y, sr = librosa.load(songname, duration=60)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

        def ensure_single_value(feature, feature_name):
            if isinstance(feature, np.ndarray) and feature.size > 1:
                print(f"Feature '{feature_name}' is an array with more than one element (size={feature.size}): {feature}")
            return np.mean(feature) if isinstance(feature, np.ndarray) else feature

        features = {
            'tempo': ensure_single_value(tempo, 'tempo'),
            'total_beats': len(beats) if beats is not None else 0,
            'average_beats': np.mean(beats) if beats is not None and len(beats) > 0 else 0,
            'chroma_stft_mean': ensure_single_value(librosa.feature.chroma_stft(y=y, sr=sr), 'chroma_stft_mean'),
            'chroma_stft_std': ensure_single_value(np.std(librosa.feature.chroma_stft(y=y, sr=sr)), 'chroma_stft_std'),
            'chroma_stft_var': ensure_single_value(np.var(librosa.feature.chroma_stft(y=y, sr=sr)), 'chroma_stft_var'),
            'chroma_cq_mean': ensure_single_value(librosa.feature.chroma_cqt(y=y, sr=sr), 'chroma_cq_mean'),
            'chroma_cq_std': ensure_single_value(np.std(librosa.feature.chroma_cqt(y=y, sr=sr)), 'chroma_cq_std'),
            'chroma_cq_var': ensure_single_value(np.var(librosa.feature.chroma_cqt(y=y, sr=sr)), 'chroma_cq_var'),
            'chroma_cens_mean': ensure_single_value(librosa.feature.chroma_cens(y=y, sr=sr), 'chroma_cens_mean'),
            'chroma_cens_std': ensure_single_value(np.std(librosa.feature.chroma_cens(y=y, sr=sr)), 'chroma_cens_std'),
            'chroma_cens_var': ensure_single_value(np.var(librosa.feature.chroma_cens(y=y, sr=sr)), 'chroma_cens_var'),
            'melspectrogram_mean': ensure_single_value(librosa.feature.melspectrogram(y=y, sr=sr), 'melspectrogram_mean'),
            'melspectrogram_std': ensure_single_value(np.std(librosa.feature.melspectrogram(y=y, sr=sr)), 'melspectrogram_std'),
            'melspectrogram_var': ensure_single_value(np.var(librosa.feature.melspectrogram(y=y, sr=sr)), 'melspectrogram_var'),
            'mfcc_mean': ensure_single_value(librosa.feature.mfcc(y=y, sr=sr), 'mfcc_mean'),
            'mfcc_std': ensure_single_value(np.std(librosa.feature.mfcc(y=y, sr=sr)), 'mfcc_std'),
            'mfcc_var': ensure_single_value(np.var(librosa.feature.mfcc(y=y, sr=sr)), 'mfcc_var'),
            'mfcc_delta_mean': ensure_single_value(librosa.feature.delta(librosa.feature.mfcc(y=y, sr=sr)), 'mfcc_delta_mean'),
            'mfcc_delta_std': ensure_single_value(np.std(librosa.feature.delta(librosa.feature.mfcc(y=y, sr=sr))), 'mfcc_delta_std'),
            'mfcc_delta_var': ensure_single_value(np.var(librosa.feature.delta(librosa.feature.mfcc(y=y, sr=sr))), 'mfcc_delta_var'),
            'rmse_mean': ensure_single_value(librosa.feature.rms(y=y), 'rmse_mean'),
            'rmse_std': ensure_single_value(np.std(librosa.feature.rms(y=y)), 'rmse_std'),
            'rmse_var': ensure_single_value(np.var(librosa.feature.rms(y=y)), 'rmse_var'),
            'cent_mean': ensure_single_value(librosa.feature.spectral_centroid(y=y, sr=sr), 'cent_mean'),
            'cent_std': ensure_single_value(np.std(librosa.feature.spectral_centroid(y=y, sr=sr)), 'cent_std'),
            'cent_var': ensure_single_value(np.var(librosa.feature.spectral_centroid(y=y, sr=sr)), 'cent_var'),
            'spec_bw_mean': ensure_single_value(librosa.feature.spectral_bandwidth(y=y, sr=sr), 'spec_bw_mean'),
            'spec_bw_std': ensure_single_value(np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr)), 'spec_bw_std'),
            'spec_bw_var': ensure_single_value(np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)), 'spec_bw_var'),
            'contrast_mean': ensure_single_value(librosa.feature.spectral_contrast(S=np.abs(librosa.stft(y)), sr=sr), 'contrast_mean'),
            'contrast_std': ensure_single_value(np.std(librosa.feature.spectral_contrast(S=np.abs(librosa.stft(y)), sr=sr)), 'contrast_std'),
            'contrast_var': ensure_single_value(np.var(librosa.feature.spectral_contrast(S=np.abs(librosa.stft(y)), sr=sr)), 'contrast_var'),
            'rolloff_mean': ensure_single_value(librosa.feature.spectral_rolloff(y=y, sr=sr), 'rolloff_mean'),
            'rolloff_std': ensure_single_value(np.std(librosa.feature.spectral_rolloff(y=y, sr=sr)), 'rolloff_std'),
            'rolloff_var': ensure_single_value(np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)), 'rolloff_var'),
            'poly_mean': ensure_single_value(librosa.feature.poly_features(y=y, sr=sr), 'poly_mean'),
            'poly_std': ensure_single_value(np.std(librosa.feature.poly_features(y=y, sr=sr)), 'poly_std'),
            'poly_var': ensure_single_value(np.var(librosa.feature.poly_features(y=y, sr=sr)), 'poly_var'),
            'tonnetz_mean': ensure_single_value(librosa.feature.tonnetz(y=y, sr=sr), 'tonnetz_mean'),
            'tonnetz_std': ensure_single_value(np.std(librosa.feature.tonnetz(y=y, sr=sr)), 'tonnetz_std'),
            'tonnetz_var': ensure_single_value(np.var(librosa.feature.tonnetz(y=y, sr=sr)), 'tonnetz_var'),
            'zcr_mean': ensure_single_value(librosa.feature.zero_crossing_rate(y), 'zcr_mean'),
            'zcr_std': ensure_single_value(np.std(librosa.feature.zero_crossing_rate(y)), 'zcr_std'),
            'zcr_var': ensure_single_value(np.var(librosa.feature.zero_crossing_rate(y)), 'zcr_var'),
            'harm_mean': ensure_single_value(librosa.effects.harmonic(y), 'harm_mean'),
            'harm_std': ensure_single_value(np.std(librosa.effects.harmonic(y)), 'harm_std'),
            'harm_var': ensure_single_value(np.var(librosa.effects.harmonic(y)), 'harm_var'),
            'perc_mean': ensure_single_value(librosa.effects.percussive(y), 'perc_mean'),
            'perc_std': ensure_single_value(np.std(librosa.effects.percussive(y)), 'perc_std'),
            'perc_var': ensure_single_value(np.var(librosa.effects.percussive(y)), 'perc_var'),
            'frame_mean': ensure_single_value(y, 'frame_mean'),
            'frame_std': ensure_single_value(np.std(y), 'frame_std'),
            'frame_var': ensure_single_value(np.var(y), 'frame_var')
        }
        return features

    def extract_label_from_filename(self, filename):
        try:
            emotion = filename.split('_')[0]
            label = self.emotion_map.get(emotion, -1)
        except IndexError:
            emotion, label = 'unknown', -1
        
        return label, emotion

    def update_database(self):
        if os.path.exists(self.database_csv):
            existingdata_df = pd.read_csv(self.database_csv)
        else:
            existingdata_df = pd.DataFrame(columns=[
                'id', 'song_name', 'class', 'label',
                'tempo', 'total_beats', 'average_beats', 'chroma_stft_mean', 'chroma_stft_std', 'chroma_stft_var',
                'chroma_cq_mean', 'chroma_cq_std', 'chroma_cq_var', 'chroma_cens_mean', 'chroma_cens_std', 'chroma_cens_var',
                'melspectrogram_mean', 'melspectrogram_std', 'melspectrogram_var', 'mfcc_mean', 'mfcc_std', 'mfcc_var',
                'mfcc_delta_mean', 'mfcc_delta_std', 'mfcc_delta_var', 'rmse_mean', 'rmse_std', 'rmse_var', 'cent_mean',
                'cent_std', 'cent_var', 'spec_bw_mean', 'spec_bw_std', 'spec_bw_var', 'contrast_mean', 'contrast_std',
                'contrast_var', 'rolloff_mean', 'rolloff_std', 'rolloff_var', 'poly_mean', 'poly_std', 'poly_var',
                'tonnetz_mean', 'tonnetz_std', 'tonnetz_var', 'zcr_mean', 'zcr_std', 'zcr_var', 'harm_mean', 'harm_std',
                'harm_var', 'perc_mean', 'perc_std', 'perc_var', 'frame_mean', 'frame_std', 'frame_var'
            ])

        new_files_data = []

        file_names = [filename for filename in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, filename))]
        id_count = existingdata_df['id'].max() + 1 if not existingdata_df.empty else 0

        for filename in file_names:
            if filename.endswith(('.wav', '.mp3')):
                songname = os.path.join(self.path, filename)
                if 'song_name' in existingdata_df.columns and filename in existingdata_df['song_name'].values:
                    continue

                features = self.extract_features_from_file(songname)
                features['song_name'] = filename

                label, emotion = self.extract_label_from_filename(filename)
                features['class'] = emotion
                features['label'] = label
                features['id'] = id_count
                id_count += 1

                if self.model:
                    # Exclude 'id', 'song_name', 'class', 'label' fields from feature_vector
                    feature_vector = [features[key] for key in features.keys() if key not in ['id', 'song_name', 'class', 'label']]

                    # Ensure the feature_vector has correct length
                    expected_num_features = self.model.n_features_in_ 
                    if len(feature_vector) != expected_num_features:
                        raise ValueError(f"Expected {expected_num_features} features, but got {len(feature_vector)} features.") 
                    
                    # Convert all items to scalar values if necessary
                    feature_vector = [f.item() if isinstance(f, (np.ndarray, np.generic)) else f for f in feature_vector]
                    
                    print(f"Extracted feature_vector for {filename}: {feature_vector}")
                    predicted_label = self.model.predict([feature_vector])[0]
                    features['label'] = predicted_label
                    predicted_emotion = [k for k, v in self.emotion_map.items() if v == predicted_label][0]
                    features['class'] = predicted_emotion

                new_files_data.append(features)
                id_count += 1

        if new_files_data:
            new_data_df = pd.DataFrame(new_files_data)
            updated_data = pd.concat([existingdata_df, new_data_df], ignore_index=True)
            updated_data.to_csv(self.database_csv, index=False)
            print("Database updated successfully.")
        else:
            print("No new files to process.")