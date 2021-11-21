# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 20:31:05 2021

@author: Vadim
"""

import pandas as pd

import pathlib 
from pathlib import Path

from sklearn import preprocessing
import joblib

neigh_from_joblib = joblib.load('telecom_classify_KNN_model.pkl')
print(neigh_from_joblib)

work_path = pathlib.Path.cwd()
data_path = Path(work_path, 'teleCust.csv')

telec_df = pd.read_csv(data_path)
telec_df1 = telec_df.drop('custcat', axis = 1)

print(telec_df1.info())

X = telec_df1[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float)) # Normalization (standartization)
yhat=neigh_from_joblib.predict(X) #prediction

print(yhat)

telec_predicted = telec_df1.assign(custcat = yhat)
telec_predicted.to_csv('predictec_customer_category_1.csv')