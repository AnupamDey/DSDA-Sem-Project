# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:37:41 2019

@author: Anupam Shankar Dey
"""

import numpy as np
import pandas as pd
#from __future__ import print_function
import matplotlib.pyplot as plt

ds = pd.read_csv('eighthr.csv')
labels = pd.read_csv('eighthrnames.csv')
labels2 = labels.iloc[:,:1].values
labeldf = pd.DataFrame(labels2)
X = ds.iloc[:,1:73].values
y = ds.iloc[:,73:].values
ds4 = pd.DataFrame(X)

cols = [i.strip() for i in labeldf[0]]

X = ds4.replace(to_replace='?',value=np.nan)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0,verbose=0)
imputer = imputer.fit(X)
X = imputer.transform(X)

X = pd.DataFrame(X,columns=cols[1:])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
y_train,y_test = y_train.ravel(),y_test.ravel()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Accuracy: ", ((cm[0][0]+cm[1][1])/len(y_test))*100)
