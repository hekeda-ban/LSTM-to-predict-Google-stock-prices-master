# _*__coding:utf-8 _*__
# @Time :2022/11/20 0020 20:40
# @Author :bay
# @File youtobe_main.py
# @Software : PyCharm
import youtobelstm.youtubelstm as ylstm

# dataset_train,cols, X_train, y_train, sc_predict, datelist_train
# ylstm.prepare_data()
prepare_data = ylstm.prepare_data()
model = ylstm.model_lstm(prepare_data[0],prepare_data[2], prepare_data[3])
PREDICTIONS_FUTURE, PREDICTION_TRAIN = ylstm.model_predict(model, prepare_data[2], prepare_data[4], prepare_data[5])
# PREDICTION_TRAIN
ylstm.plot(prepare_data[0], prepare_data[1], prepare_data[5], PREDICTIONS_FUTURE, PREDICTION_TRAIN)
#

