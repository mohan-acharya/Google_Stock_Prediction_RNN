#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 01:11:03 2018

@author: mohanacharya
"""

# RNN for predicting Google Stock Prices 

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import only the training set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# numpy array that is input to the neural network
training_set = dataset_train.iloc[:,1:2].values

# Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
# creating a object of class
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# create a data structure wuth 60 timesteps and 1 output
  #  timesteps is the number of past outputs needed to predict next output */
  
X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train = np.array(X_train),np.array(y_train)

# Reshaping the data ( adding dimensionality)
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

# PART - 2 Building the RNN 

# Importing the Keras Libraries and Packages
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout

# initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout Regularisation
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout Regularisation 
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout Regularisation
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout Regularisation
regressor.add(LSTM(units=50,return_sequences=False))
regressor.add(Dropout(0.2))

# Adding the OUTPUT layer
regressor.add(Dense(units=1))

# compiling the RNN 
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the training set
regressor.fit(X_train,y_train,epochs = 100,batch_size=32)

# Making the predictions and visualising the results

#getting the real stock prices
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
# numpy array that is input to the neural network
real_stock_price = dataset_test.iloc[:,1:2].values

# getting the predicted value of stock
# concatenation of training and test set 
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
# scaling the inputs
inputs = sc.transform(inputs)
X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red',label='Real Stock Price')
plt.plot(predicted_stock_price, color = 'green',label='Predicted Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()





