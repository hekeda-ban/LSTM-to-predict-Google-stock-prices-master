# _*__coding:utf-8 _*__
# @Time :2022/11/15 0015 9:56
# @Author :bay
# @File zhihu_model.py
# @Software : PyCharm
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import numpy as np


# todo 定义将时间序列预测问题转化为监督学习问题的函数
# 输入值为历史值，输出值为预测值
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []
    # i: n_in, n_in-1, ..., 1，为滞后期数
    # 分别代表t-n_in, ... ,t-1期
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # i: 0, 1, ..., n_out-1，为超前预测的期数
    # 分别代表t，t+1， ... ,t+n_out-1期
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# todo 定义准备数据的函数
def prepare_data(filepath, n_in, n_out=30, n_vars=4, train_proportion=0.8):
    dataset = read_csv(filepath, encoding='gbk')
    # 设置时间戳索引
    dataset['日期'] = pd.to_datetime(dataset['日期'])
    dataset.set_index("日期", inplace=True)
    values = dataset.values
    # 保证所有数据都是float32类型
    values = values.astype('float32')
    # 变量归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # print("scaled:\n", scaled)
    # 将时间序列问题转化为监督学习问题
    reframed = series_to_supervised(scaled, n_in, n_out)
    # print("reframed:\n", reframed)
    # 取出保留的变量
    contain_vars = []
    for i in range(1, n_in+1):
        contain_vars += [('var%d(t-%d)' % (j, i)) for j in range(1, n_vars+1)]
    data = reframed[contain_vars + ['var1(t)'] + [('var1(t+%d)' % (j)) for j in range(1, n_out)]]
    # 修改列名
    col_names = ['Y', 'X1', 'X2', 'X3']
    contain_vars = []
    for i in range(n_vars):
        contain_vars += [('%s(t-%d)' % (col_names[i], j)) for j in range(1, n_in+1)]
    data.columns = contain_vars + ['Y(t)'] + [('Y(t+%d)' % (j)) for j in range(1, n_out)]
    # print(data.columns)
    # 分隔数据集，分为训练集和测试集
    values = data.values
    n_train = round(data.shape[0]*train_proportion)
    # print("n_train:\n", n_train)  # 784
    train = values[:n_train, :]
    # print(train)
    test = values[n_train:, :]
    # # 分隔输入X和输出y
    train_X, train_y = train[:, :n_in*n_vars], train[:, n_in*n_vars:]
    test_X, test_y = test[:, :n_in*n_vars], test[:, n_in*n_vars:]
    # 将输入X改造为LSTM的输入格式，即[samples,timesteps,features]
    train_X = train_X.reshape((train_X.shape[0], n_in, n_vars))
    test_X = test_X.reshape((test_X.shape[0], n_in, n_vars))
    # print(train_X)
    # print("train_X.shape", train_X.shape)
    # print(train_y)
    # print("train_y.shape", train_y.shape)
    return scaler, data, train_X, train_y, test_X, test_y, dataset


# todo 定义拟合LSTM模型的函数
def fit_lstm(data_prepare, n_neurons=50, n_batch=72, n_epoch=100, loss='mae', optimizer='adam', repeats=1):
    train_X = data_prepare[2]
    train_y = data_prepare[3]
    test_X = data_prepare[4]
    test_y = data_prepare[5]
    model_list = []
    for i in range(repeats):
        # 设计神经网络
        model = Sequential()
        model.add(LSTM(n_neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(train_y.shape[1]))
        model.compile(loss=loss, optimizer=optimizer)
        # 拟合神经网络
        # history = model.fit(train_X, train_y, epochs=n_epoch, batch_size=n_batch, validation_data=(test_X, test_y), verbose=0, shuffle=False)
        # 画出学习过程
        # p1 = pyplot.plot(history.history['loss'], color='blue', label='train')
        # p2 = pyplot.plot(history.history['val_loss'], color='yellow',label='test')
        # 保存model
        model_list.append(model)
    # pyplot.legend(["train", "test"])
    # pyplot.show()
    return model_list


# todo 定义预测的函数
def lstm_predict(model, data_prepare):
    scaler = data_prepare[0]
    test_X = data_prepare[4]
    test_y = data_prepare[5]
    # 做出预测
    yhat = model.predict(test_X)
    # 将测试集上的预测值还原为原来的数据维度
    scale_new = MinMaxScaler()
    scale_new.min_, scale_new.scale_ = scaler.min_[0], scaler.scale_[0]
    inv_yhat = scale_new.inverse_transform(yhat)
    # 将测试集上的实际值还原为原来的数据维度
    inv_y = scale_new.inverse_transform(test_y)
    return inv_yhat, inv_y


# todo: 定义预测评价的函数（RMSE）
# 计算每一步预测的RMSE
def evaluate_forecasts(test, forecasts, n_out):
    rmse_dic = {}
    for i in range(n_out):
        actual = [float(row[i]) for row in test]
        predicted = [float(forecast[i]) for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        rmse_dic['t+' + str(i+1) + ' RMSE'] = rmse
    return rmse_dic


# TODO 定义将预测可视化的函数
#以原始数据为背景画出预测数据
def plot_forecasts(series, forecasts):
    #用蓝色画出原始数据集
    pyplot.plot(series.values)
    n_seq = len(forecasts[0])
    #用红色画出预测值
    for i in range(1,len(forecasts)+1):
        xaxis = [x for x in range(i, i+n_seq+1)]
        yaxis = [float(series.iloc[i-1,0])] + list(forecasts[i-1])
        pyplot.plot(xaxis, yaxis, color='red')
    #展示图像
    pyplot.show()


# 建立模型
# 建立模型（n_in = 15，n_neuron = 5，n_batch = 16，n_epoch = 200）
# 为了减少随机性，重复建立五次模型，取五次结果的平均作为最后的预测。
# 定义需要的变量

#
# n_vars = 4


#
# scaler, data, train_X, train_y, test_X, test_y, dataset = data_prepare

#
#
#
#

#
# inv_yhat_ave = inv_yhat_ave / repeats
#
# rmse_dic_list = []
# for i in range(len(model_list)):
#     inv_yhat = inv_yhat_list[i]
#     inv_y = inv_y_list[i]
#     rmse_dic = evaluate_forecasts(inv_y, inv_yhat, n_out)
#     rmse_dic_list.append(rmse_dic)
#
# rmse_dic_list.append(evaluate_forecasts(inv_y, inv_yhat_ave, n_out))
#
# df_dic = {}
# for i in range(len(rmse_dic_list) - 1):
#     df_dic['第' + str(i + 1) + '次'] = pd.Series(rmse_dic_list[i])
#
# df_dic['平均'] = pd.Series(rmse_dic_list[i + 1])
# rmse_df = DataFrame(df_dic)
# rmse_df