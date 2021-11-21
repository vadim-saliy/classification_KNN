# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 22:35:15 2021

@author: Vadim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pathlib 
from pathlib import Path

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import joblib
#from joblib import dump, load


work_path = pathlib.Path.cwd()
data_path = Path(work_path, 'teleCust.csv')

telec_df = pd.read_csv(data_path)
print(telec_df.head(10))
print(telec_df.describe())
print(telec_df.info())
print(telec_df['custcat'].value_counts())

#convert pandas df to nympy array:
# features
X = telec_df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
#target (Labels)
y = telec_df['custcat'].values

#Normalization (standartization)
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float)) # Normalization (standartization)

#splitting to train/test:
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

Ks = 35 #any relevant value on our choise - neighbors quantity

mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

#Find the optimal K:
    
for n in range(1,Ks):
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

#Train Model with optimal K:
k = mean_acc.argmax()+1 
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print(neigh)

#save model neigh:
joblib.dump(neigh,'telecom_classify_KNN_model.pkl')



