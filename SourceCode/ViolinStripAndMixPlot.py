import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('Dataset/Emotion_data.csv')
feature = data.ix[:, 'tempo':]
target = data['label']
targetName = data['class']
featureName = list(feature)

for name in featureName:
    plt.figure(figsize=(12, 12))
    sns.stripplot(x='class', y=name, data=data, jitter=True)
    plt.title('Strip Plot for ' + name)
    plt.savefig('Plots\\Strip Plot\\' + name)
    plt.show()
    plt.clf()
    
    plt.figure(figsize=(12, 12))
    sns.violinplot(x='class', y=name, data=data)
    plt.title('Violin Plot for ' + name)
    plt.savefig('Plots\\Violin Plot\\' + name)
    plt.show()
    plt.clf()
    
    plt.figure(figsize=(12, 12))
    sns.violinplot(x='class', y=name, data=data, inner=None, color='lightgray')
    sns.stripplot(x='class', y=name, data=data, jitter=True)
    plt.title('Violin and Strip Plot for ' + name)
    plt.savefig('Plots\\Violin and Strip Plot\\' + name)
    plt.show()
    plt.clf()