# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 23:00:20 2019

@author: jaehooncha

Linear regression
"""
import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

train_x = np.linspace(-3, 3, num = 100).reshape(-1,1)
train_y = 1/(1+np.exp(-train_x))
    
test_x = np.linspace(3.1, 4, num = 10).reshape(-1,1)
test_y = 1/(1+np.exp(-test_x))

    
def linear_regression_train(X, Y):
    clf = LinearRegression()
    clf.fit(X, Y)
    return clf

def linear_regression_prediction(lr_model, X):
    linear_regression = lr_model
    linear_prediction = linear_regression.predict(X)
    linear_prediction = linear_prediction.reshape(-1)
    return linear_prediction


linear_regression_model = linear_regression_train(train_x, train_y)
train_predict_y = linear_regression_prediction(linear_regression_model, train_x)
test_predict_y = linear_regression_prediction(linear_regression_model, test_x)

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
