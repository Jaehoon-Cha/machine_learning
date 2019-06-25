# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 23:15:24 2019

@author: jaehooncha

Support Vector Machine 
using the Radial Basis Function (RBF) kernel for prediction
"""
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

train_x = np.linspace(-3, 3, num = 100).reshape(-1,1)
train_y = 1/(1+np.exp(-train_x))
train_y = train_y.reshape(-1)
    
test_x = np.linspace(3.1, 4, num = 10).reshape(-1,1)
test_y = 1/(1+np.exp(-test_x))
test_y = test_y.reshape(-1)    

def svm_train(X, Y):
    clf = SVR(kernel= 'rbf', gamma= 0.05)
    clf.fit(X, Y)
    return clf

def svm_prediction(svm_model, X):
    svm_regression = svm_model
    svm_prediction = svm_regression.predict(X)
    svm_prediction = svm_prediction.reshape(-1)
    return svm_prediction

svm_model = svm_train(train_x, train_y)
train_predict_y = svm_prediction(svm_model, train_x)
test_predict_y = svm_prediction(svm_model, test_x)


train_rmse = np.sqrt(np.mean((train_predict_y - train_y)**2))
test_rmse = np.sqrt(np.mean((test_predict_y - test_y)**2))
print('train RMSE is %.4f' %(train_rmse))
print('test RMSE is %.4f' %(test_rmse))
#
plt.figure(figsize=(15,7))
plt.plot(train_y, label = 'true')
plt.plot(train_predict_y, label = 'prediction')
plt.legend()

plt.figure(figsize=(15,7))
plt.plot(test_y, label = 'true')
plt.plot(test_predict_y, label = 'prediction')
plt.legend()
