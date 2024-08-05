import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.model = RandomForestClassifier(n_estimators=100)

    def train_and_evaluate(self):
        # Normaliserer funksjonene
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)
        
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        
        plt.barh(range(len(self.model.feature_importances_)), self.model.feature_importances_)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.show()
        
        # Lagre modellen til en fil
        joblib.dump(self.model, 'Emotion_Audio_Detection_Model.pkl')
        print("Model saved to Emotion_Audio_Detection_Model.pkl")


# Anta at DataProcessor-delen forbereder riktig feature-setter
# Erstatt dette med hvordan du laster inn dine data og label
# features, labels = prepare_features_and_labels()
# trainer = ModelTrainer(features, labels)
# trainer.train_and_evaluate()