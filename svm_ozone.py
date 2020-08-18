# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 08:03:15 2019

@author: Anupam Shankar Dey
"""

# Import the libraries
from __future__ import division, print_function
import numpy as np
import pandas as pd
from sklearn import datasets, svm 
from sklearn.model_selection import train_test_split
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

# Split the dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
y_train,y_test = y_train.ravel(),y_test.ravel()

# Evaluation function @ return accuracy
def evaluation_on_test_set(model=None):
    preds = model.predict(X_test)
    correct_classifications = 0
    for i in range(len(y_test)):
        if preds[i] == y_test[i]:
            correct_classifications += 1
    acc = (correct_classifications/len(y_test))*100
    return acc

# Fit the model and obtain the accuracies for different kernels : Try it on your system by uncommenting the below lines for seperate kernels
# Mine takes forever to compute for Gaussian and polynomial kernels.
from sklearn.metrics import confusion_matrix
# kernels = ('linear','poly','rbf')
acc_list = []
cm_list = []
#for i,kernel in enumerate(kernels):
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train,y_train)
accuracy = evaluation_on_test_set(svm_model)
acc_list.append(accuracy)
print("{} accuracy using the linear kernel".format(accuracy))
cm = confusion_matrix(y_test,svm_model.predict(X_test))
cm_list.append(cm)

# Visualisation of results
linear_svc = svm.SVC(kernel='linear').fit(X_train,y_train)
# poly_svc = svm.SVC(kernel='poly',degree=3).fit(X_train,y_train)
# rbf_svc = svm.SVC(kernel='rbf',gamma=0.7).fit(X_train,y_train)

# Displaying the corresponding support vectors
#for i,clf in enumerate((linear_svc, rbf_svc, poly_svc)):
#    print("The support vectors of {} classifier are {}".format(clf,clf.support_vectors_),end='\n')

#h = 0.02
#X_min , X_max = X[:,0].min() - 1,X[:,0].max() + 1
#y_min , y_max = X[:,1].min() - 1,X[:,1].max() + 1
#xx , yy = np.meshgrid(np.arange(X_min,X_max,h),np.arange(y_min,y_max,h))

# titles = ['SVC with linear kernel',
#           'SVC with RBF kernel',
#           'SVC with polynomial (degree 3) kernel']
#titles = ['SVC with linear kernel']

# for i, clf in enumerate((linear_svc, rbf_svc, poly_svc)):
#     plt.figure(i)
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.ocean)
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.xticks(())
#     plt.yticks(())
#     plt.title(titles[i])

# plt.show()