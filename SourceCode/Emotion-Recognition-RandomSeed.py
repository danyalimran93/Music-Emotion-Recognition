"""
@author: Danyal

The following code classifies piece of music as one of 
the four emotions mentioned in the document
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from pandas.tools.plotting import scatter_matrix

s

data = pd.read_csv('Dataset/Emotion_data.csv')
feature = data.ix[:, 'tempo':]
labels = list(feature)
color = ['red' if l==1 else 'green' if l==2 else 'blue' if l==3 else 'orange' for l in data['label']]

plt.style.use('ggplot')

array = np.array(data)

result = []
xlabel = []
color = []
colors = ['red', 'green', 'blue']
index = 0

for random_seed in range(1, 11):
    features = array[:, 5:]
    labels = data.ix[:, 'class'].dropna()
    test_size = 0.30
    
    train_d, test_d, train_l, test_l = train_test_split(features, labels, test_size=test_size, random_state=random_seed)

    for neighbors in range(1, 10):
        kNN = KNeighborsClassifier(n_neighbors=neighbors)
        kNN.fit(train_d, train_l)
        prediction = kNN.predict(test_d)
        xlabel.append(neighbors)
        result.append(accuracy_score(prediction, test_l))
        color.append(colors[index])
        index = (index+1)%3

plt.figure(figsize=(10, 10))
plt.xlabel('kNN Neighbors for k=1,2...10')
plt.ylabel('Accuracy Score')
plt.title('kNN Classifier Results')
plt.ylim(0, 1)
plt.scatter(xlabel, result, color=color)
plt.savefig('10-folds kNN Result.png')
plt.show()