# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:33:00 2019

@author: jaehooncha

Multi-Layer Perceptron
"""
import tensorflow as tf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

### data load ###
train = pd.read_pickle('../datasets/sinusoid_train.pickle')
test = pd.read_pickle('../datasets/sinusoid_test.pickle')


### seperate features and target ###
train_x = np.array(train[0]).reshape(-1,1)
train_y = np.array(train[1]).reshape(-1,1)

test_x = np.array(test[0]).reshape(-1,1)
test_y = np.array(test[1]).reshape(-1,1)  


### normal ###
max_y = np.max(train_y)
min_y = np.min(train_y)

train_y = 2*(train_y-min_y)/(max_y-min_y)-1
test_y = 2*(test_y-min_y)/(max_y-min_y)-1


def mlp_train(X, Y, h1 = 100, h2 = 50, h3 = 5, Ep = 100, Bat = 100, Lr = 0.005):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape = (X.shape[1:])))
    model.add(tf.keras.layers.Dense(h1, activation = tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(h2, activation = tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(h3, activation = tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.Adam(lr=Lr)
    
    model.compile(optimizer = optimizer,
                 loss = 'mse',
                 metrics =['mae', 'mse'])

    hist = model.fit(X, Y, epochs = Ep, batch_size = Bat)   
    return hist

def mlp_predict(mlp_model, X):
    mlp = mlp_model
    mlp_prediction = mlp.predict(X)
    mlp_prediction = mlp_prediction.reshape(-1)
    return mlp_prediction


### implement ###
mlp_model = mlp_train(train_x, train_y, Ep = 250, Bat = 100).model
train_predict_y = mlp_predict(mlp_model, train_x)
test_predict_y = mlp_predict(mlp_model, test_x)


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

