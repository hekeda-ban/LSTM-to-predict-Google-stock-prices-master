# _*__coding:utf-8 _*__
# @Time :2022/11/16 0016 10:04
# @Author :bay
# @File youtubelstm.py
# @Software : PyCharm

# todo PART1 data pre-processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

n_future = 60
n_past = 90


def prepare_data():
    # step1 Read data
    # ['Open', 'High', 'Low', 'Close', 'Volume']
    dataset_train = pd.read_csv('../Google_Stock_Price_Train.csv')
    cols = list(dataset_train)[1:6]

    datelist_train = list(dataset_train['Date'])
    # datelist_train = [dt.datetime.strptime(date, '%m/%d/%Y').date() for date in datelist_train]
    # 数据时间列处理
    new_dates = [date.split('/')[2] + '-' + date.split('/')[0] + '-' + date.split('/')[1] for date in datelist_train ]
    datelist_train = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in new_dates]
    print("Training set shape =={}".format(dataset_train.shape))   # Training set shape ==(1258, 6)
    print("All timestamps =={}".format(len(datelist_train)))   # All timestamps ==1258
    print("featured selected:{}".format(cols))
    #
    # step2 data pre-processing(shapping transformations)
    dataset_train = dataset_train[cols].astype(str)
    # 数据处理 将,去掉
    for i in cols:
        for j in range(0, len(dataset_train)):
            dataset_train[i][j] = dataset_train[i][j].replace(',', '')
    dataset_train = dataset_train.astype(float)
    # array类型 这里还需要修改
    training_set = dataset_train.values
    print("shape of trianing set == {}".format(training_set.shape))
    # print(training_set)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    training_set_scaled = sc.fit_transform(training_set)
    # print(training_set_scaled)

    sc_predict = StandardScaler()
    sc_predict.fit_transform(training_set[:, 0:1])
    X_train = []
    y_train = []
    for i in range(n_past, len(training_set_scaled)-n_future + 1):
        X_train.append(training_set_scaled[i-n_past:i, 0:dataset_train.shape[1]-1])
        y_train.append(training_set_scaled[i+n_future-1:i+n_future, 0])

    # X_train[]  90x4 每次下移一行
    # y_train 是个[]

    X_train, y_train = np.array(X_train), np.array(y_train)
    # print(X_train)
    # print("y_train:\n", y_train)
    print("X_train shape =={}".format(X_train.shape))   # (1109, 90, 4)
    print("y_train shape =={}".format(y_train.shape))   # (1109, 1)
    # print(y_train)

    return dataset_train, cols, X_train, y_train, sc_predict, datelist_train


def model_lstm(dataset_train, X_train, y_train):
    # todo PART2 create a model training
    # building-up thr lstm based Neural Network
    # print(n_past, dataset_train.shape[1]-1)
    # input_shape = (90, 4)
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, dataset_train.shape[1]-1)))
    model.add(LSTM(units=10, return_sequences=False))
    model.add(Dropout(0.25))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    # start training
    es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
    mcp = ModelCheckpoint(filepath='../weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    tb = TensorBoard('../logs')
    history = model.fit(X_train, y_train, shuffle=True, epochs=30, callbacks=[es, rlr, mcp, tb], validation_split=0.2,
                        verbose=1, batch_size=256)

    print("history:\n", history)
    return model


def datetime_to_timestamp(x):
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


def model_predict(model, X_train, sc_predict, datelist_train):
    # todo PART3 future prediction
    # make predictions for future date
    # 报错 id也没有呀
    # datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1d').tolist()
    datelist_future = pd.date_range(datelist_train[-1], periods=n_future).tolist()

    prediction_future = model.predict(X_train[-n_future:])
    y_pred_future = sc_predict.inverse_transform(prediction_future)
    PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Open']).set_index(pd.Series(datelist_future))
    # print(PREDICTIONS_FUTURE.head(3))
    #  TODO 新增的
    # datelist_future_ = []
    # for this_timestamp in datelist_future:
    #     datelist_future_.append(this_timestamp.date())

    prediction_train = model.predict(X_train[n_past:])
    y_pred_train = sc_predict.inverse_transform(prediction_train)
    PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Open']).set_index(pd.Series(datelist_train[2*n_past+n_future-1:]))
    # PREDICTIONS_TRAIN = pd.DataFrame(y_pred_train, columns=['Open']).set_index(pd.Series(datelist_future_))
    PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(datetime_to_timestamp)
    print(" PREDICTION_TRAIN.index:\n", PREDICTION_TRAIN.index)
    print("PREDICTIONS_FUTURE:\n", PREDICTIONS_FUTURE.head(3), PREDICTIONS_FUTURE.shape)  # (60, 1)
    print("PREDICTIONS_TRAIN:\n", PREDICTION_TRAIN.head(3), PREDICTION_TRAIN.shape)  # (1019, 1)
    return PREDICTIONS_FUTURE, PREDICTION_TRAIN


# visualize the predictions
def plot(dataset_train, cols, datelist_train, PREDICTIONS_FUTURE, PREDICTION_TRAIN):
    from pylab import rcParams
    rcParams['figure.figsize'] = 14, 5
    dataset_train = pd.DataFrame(dataset_train, columns=cols)
    dataset_train.index = datelist_train
    dataset_train.index = pd.to_datetime(dataset_train.index)
    # print("dataset_train:\n", dataset_train)
    print("dataset_train.index:\n", dataset_train.index)

    # print("dataset_train.index:\n", dataset_train.index)
    print("PREDICTIONS_FUTURE.index:\n", PREDICTIONS_FUTURE.index)
    print("最小值", min(PREDICTIONS_FUTURE.index))
    Start_date_for_plotting = '2012/12/14'
    # print("print(dataset_train.loc[Start_date_for_plotting:].index):\n", dataset_train.loc[Start_date_for_plotting:].index)
    plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Open'], color='r', label='Predicted stock price')
    # 新加的
    # ValueError: x and y must have same first dimension, but have shapes (1019,) and (1258,)
    plt.plot(PREDICTION_TRAIN.loc[Start_date_for_plotting:].index, PREDICTION_TRAIN.loc[Start_date_for_plotting:]['Open'],
             color='orange', label='Training predictions')
    plt.plot(dataset_train.loc[Start_date_for_plotting:].index, dataset_train.loc[Start_date_for_plotting:]['Open'],
             color='b', label='Actual stock price')
    plt.axvline(x=min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')
    plt.grid(which='major', color='#cccccc', alpha=0.5)
    plt.legend(shadow=True)
    plt.title('Predicted and Actual stock price', family='Arial', fontsize=12)
    plt.xlabel('Timeline', family='Arial', fontsize=10)
    plt.ylabel('stock price value', family='Arial', fontsize=10)
    plt.xticks(rotation=45, fontsize=5)
    plt.show()