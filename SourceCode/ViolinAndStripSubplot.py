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
    
    plt.subplot(2,1,1)
    sns.stripplot(x='class', y=name, data=data, jitter=True)
    plt.title('Strip Plot for ' + name)
    
    plt.subplot(2,1,2)
    sns.violinplot(x='class', y=name, data=data)
    plt.title('Violin Plot for ' + name)
    
    plt.tight_layout()
    plt.savefig('Plots\\Violin and Strip Subplot\\' + name)
    plt.show()
    plt.clf()