import pandas as pd

class DataProcessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load_and_normalize_data(self):
        dataframe = pd.read_csv(self.csv_path)

        # Extract numerical features for normalization
        feature_columns = dataframe.columns.difference(['label', 'song_name', 'index', 'id', 'class'])
        numeric_feature_columns = dataframe[feature_columns].select_dtypes(include=['float64', 'int64']).columns
        feature = dataframe[numeric_feature_columns]

        # Normalize features
        for name in feature.columns:
            feature[name] = (feature[name] - feature[name].min()) / (feature[name].max() - feature[name].min())

        features = feature.values
        labels = dataframe['label'].values

        return features, labels