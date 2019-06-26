# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 23:15:24 2019

@author: jaehooncha

Support Vector Machine 
using the Radial Basis Function (RBF) kernel for prediction
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt


### data load ###
train = pd.read_pickle('../datasets/train_dataset.pickle')
test = pd.read_pickle('../datasets/test_dataset.pickle')


### seperate features and target ###
train_x = np.array(train[0]).reshape(-1,1)
train_y = np.array(train[1]).reshape(-1,1)

test_x = np.array(test[0]).reshape(-1,1)
test_y = np.array(test[1]).reshape(-1,1)  


### SVM ###
def svm_train(X, Y):
    clf = SVR(kernel= 'rbf', gamma= 0.05)
    clf.fit(X, Y)
    return clf

def svm_prediction(svm_model, X):
    svm_regression = svm_model
    svm_prediction = svm_regression.predict(X)
    svm_prediction = svm_prediction.reshape(-1)
    return svm_prediction


### implement ###
svm_model = svm_train(train_x, train_y)
train_predict_y = svm_prediction(svm_model, train_x)
test_predict_y = svm_prediction(svm_model, test_x)


### root mean squared error ###
train_rmse = np.sqrt(np.mean((train_predict_y - train_y)**2))
test_rmse = np.sqrt(np.mean((test_predict_y - test_y)**2))
print('train RMSE is %.4f' %(train_rmse))
print('test RMSE is %.4f' %(test_rmse))


### font size ###
plt.rcParams.update({'font.size': 15})


### draw outputs ###
plt.figure(figsize=(15,7))
plt.scatter(train_x, train_y, label = 'true', c = 'k')
plt.scatter(train_x, train_predict_y, label = 'prediction', c = 'r')
plt.xlabel('X', size = 20)
plt.ylabel('Y', size = 20)
plt.legend()

plt.figure(figsize=(15,7))
plt.scatter(test_x, test_y, label = 'true', c = 'k')
plt.scatter(test_x, test_predict_y, label = 'prediction', c = 'r')
plt.xlabel('X', size = 20)
plt.ylabel('Y', size = 20)
plt.legend()
