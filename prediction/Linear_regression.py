# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 23:00:20 2019

@author: jaehooncha

Linear regression
"""
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

### data load ###
train = pd.read_pickle('../datasets/sinusoid_train.pickle')
test = pd.read_pickle('../datasets/sinusoid_test.pickle')


### seperate features and target ###
train_x = np.array(train[0]).reshape(-1,1)
train_y = np.array(train[1]).reshape(-1,1)

test_x = np.array(test[0]).reshape(-1,1)
test_y = np.array(test[1]).reshape(-1,1)  


### Linear regression ###
def linear_regression_train(X, Y):
    clf = LinearRegression()
    clf.fit(X, Y)
    return clf

def linear_regression_prediction(lr_model, X):
    linear_regression = lr_model
    linear_prediction = linear_regression.predict(X)
    linear_prediction = linear_prediction.reshape(-1)
    return linear_prediction


### implement ###
linear_regression_model = linear_regression_train(train_x, train_y)
train_predict_y = linear_regression_prediction(linear_regression_model, train_x)
test_predict_y = linear_regression_prediction(linear_regression_model, test_x)


### root mean squared error ###
train_rmse = np.sqrt(np.mean((train_predict_y - train_y)**2))
test_rmse = np.sqrt(np.mean((test_predict_y - test_y)**2))
print('train RMSE is %.4f' %(train_rmse))
print('test RMSE is %.4f' %(test_rmse))


### font size ###
plt.rcParams.update({'font.size': 15})


### draw outputs ###
plt.figure(figsize=(15,7))
plt.scatter(train_x, train_y, label = 'true', c ='k')
plt.scatter(train_x, train_predict_y, label = 'prediction', c = 'r')
plt.xlabel('X', size = 20)
plt.ylabel('Y', size = 20)
plt.legend(loc = 1)

plt.figure(figsize=(15,7))
plt.scatter(test_x, test_y, label = 'true', c = 'k')
plt.scatter(test_x, test_predict_y, label = 'prediction', c = 'r')
plt.xlabel('X', size = 20)
plt.ylabel('Y', size = 20)
plt.legend(loc = 1)
