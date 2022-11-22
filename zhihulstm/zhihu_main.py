# _*__coding:utf-8 _*__
# @Time :2022/11/18 0018 16:41
# @Author :bay
# @File zhihu_main.py
# @Software : PyCharm
import zhihulstm.zhihu_model as zm
import numpy as np

filepath = './data.csv'
n_in = 15
n_out = 30
n_neuron = 5
n_batch = 16
n_epoch = 200
repeats = 5
# zm.prepare_data(filepath, n_in, n_out)
# train_X (784, 15, 4)
inv_yhat_list = []
inv_y_list = []
data_prepare = zm.prepare_data(filepath, n_in, n_out)
model_list = zm.fit_lstm(data_prepare, n_neuron, n_batch, n_epoch, repeats=repeats)
for i in range(len(model_list)):
    model = model_list[i]
    inv_yhat = zm.lstm_predict(model, data_prepare)[0]
    inv_y = zm.lstm_predict(model, data_prepare)[1]
    inv_yhat_list.append(inv_yhat)
    inv_y_list.append(inv_y)
    inv_yhat_ave = np.zeros(inv_y.shape)
    for i in range(repeats):
        inv_yhat_ave += inv_yhat_list[i]