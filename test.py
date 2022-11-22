# _*__coding:utf-8 _*__
# @Time :2022/11/15 0015 16:33
# @Author :bay
# @File test.py
# @Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

training_set = pd.read_csv('Google_Stock_Price_Train.csv')
# 取Open列的数据
# 我们将关注公开股票价格，并预测2017年1月的价格
training_set = training_set.iloc[:, 1:2].values
# print(training_set)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
print(training_set)
# 训练集包含1258个值，因此输入应限制为1257，另一方面，输出不能包含第0天的预测
X_train = training_set[0:1257]
print(X_train)
y_train = training_set[1:1258]
print(y_train)
X_train = np.reshape(X_train, (1257, 1, 1))
print(X_train)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
regressor = Sequential()
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, y_train, batch_size=32, epochs=200)

# test_set = pd.read_csv('Google_Stock_Price_Test.csv')
# real_stock_price = test_set.iloc[:, 1:2].values
#
# inputs = real_stock_price
# inputs = sc.transform(inputs)
# inputs = np.reshape(inputs, (20, 1, 1))
# predicted_stock_price = regressor.predict(inputs)
# predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#
# plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
# plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
# plt.title('Google Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Google Stock Price')
# plt.legend()
# plt.show()

real_stock_price_train = pd.read_csv('Google_Stock_Price_Train.csv')
real_stock_price_train = real_stock_price_train.iloc[:, 1:2].values

# Getting the predicted stock price of 2012 - 2016
predicted_stock_price_train = regressor.predict(X_train)
predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)

# Visualising the results
plt.plot(real_stock_price_train, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price_train, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
