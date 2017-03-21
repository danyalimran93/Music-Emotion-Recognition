import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv('Dataset/Emotion_data.csv')
X = data.ix[:, 'tempo':]
y = data['class']
featureName = list(X)

for name in featureName:
    X[name] = (X[name]-X[name].min())/(X[name].max()-X[name].min())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=26)

knn = KNeighborsClassifier()
param_grid = { 'n_neighbors': np.arange(1, 25) }
knn_cv = GridSearchCV(knn, param_grid, cv=10)
knn_cv.fit(X_train, y_train)

print(knn_cv.best_params_)
print("Baseline Accuracy: "),
print(knn_cv.best_score_)

y_pred = knn_cv.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print("Testing Accuracy: "),
print(accuracy_score(y_test, y_pred))