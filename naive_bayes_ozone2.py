# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 02:18:03 2019

@author: Anupam Shankar Dey
"""

from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
y_train,y_test = y_train.ravel(),y_test.ravel()

#Making likelihood estimations

#Find the two classes
X_train_class_0 = [X_train[i] for i in range(len(X_train)) if y_train[i]==0]
X_train_class_1 = [X_train[i] for i in range(len(X_train)) if y_train[i]==1]

#Find the class specific likelihoods of each feature
likelihoods_class_0 = np.mean(X_train_class_0, axis=0)/100.0
likelihoods_class_1 = np.mean(X_train_class_1, axis=0)/100.0

#Calculate the class priors
num_class_0 = float(len(X_train_class_0))
num_class_1 = float(len(X_train_class_1))

prior_probability_class_0 = num_class_0 / (num_class_0 + num_class_1)
prior_probability_class_1 = num_class_1 / (num_class_0 + num_class_1)

log_prior_class_0 = np.log10(prior_probability_class_0)
log_prior_class_1 = np.log10(prior_probability_class_1)

def calculate_log_likelihoods_with_naive_bayes(feature_vector, Class):
    assert len(feature_vector) == num_features
    log_likelihood = 0.0 #using log-likelihood to avoid underflow
    if Class==0:
        for feature_index in range(len(feature_vector)):
            if feature_vector[feature_index] == 1: #feature present
                log_likelihood += np.log10(likelihoods_class_0[feature_index]) 
            elif feature_vector[feature_index] == 0: #feature absent
                log_likelihood += np.log10(1.0 - likelihoods_class_0[feature_index])
    elif Class==1:
        for feature_index in range(len(feature_vector)):
            if feature_vector[feature_index] == 1: #feature present
                log_likelihood += np.log10(likelihoods_class_1[feature_index]) 
            elif feature_vector[feature_index] == 0: #feature absent
                log_likelihood += np.log10(1.0 - likelihoods_class_1[feature_index])
    else:
        raise ValueError("Class takes integer values 0 or 1")
        
    return log_likelihood

def calculate_class_posteriors(feature_vector):
    log_likelihood_class_0 = calculate_log_likelihoods_with_naive_bayes(feature_vector, Class=0)
    log_likelihood_class_1 = calculate_log_likelihoods_with_naive_bayes(feature_vector, Class=1)
    
    log_posterior_class_0 = log_likelihood_class_0 + log_prior_class_0
    log_posterior_class_1 = log_likelihood_class_1 + log_prior_class_1
    
    return log_posterior_class_0, log_posterior_class_1

def classify_day(document_vector):
    feature_vector = [int(element>0.0) for element in document_vector]
    log_posterior_class_0, log_posterior_class_1 = calculate_class_posteriors(feature_vector)
    if log_posterior_class_0 > log_posterior_class_1:
        return 0
    else:
        return 1
    
#Predict ozone day or not on the test set
predictions = []
for day in X_test:
    predictions.append(classify_day(day))
    
def evaluate_performance(predictions, ground_truth_labels):
    correct_count = 0.0
    for item_index in xrange(len(predictions)):
        if predictions[item_index] == ground_truth_labels[item_index]:
            correct_count += 1.0
    accuracy = correct_count/len(predictions)
    return accuracy

accuracy_of_naive_bayes = evaluate_performance(predictions, y_test)
print(accuracy_of_naive_bayes)

#for i in range(100):
#    print(predictions[i], y_test[i])