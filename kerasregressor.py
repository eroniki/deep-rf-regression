#!/usr/bin/python
import os
import time
import signal
import sys
import shutil

import wifi_data

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adagrad, RMSprop, Adam
from keras import backend as K
from keras.regularizers import l2, activity_l2
from keras.callbacks import Callback, TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor

import numpy as np
from matplotlib import pyplot as plt

def localization_loss(y, y_hat):
    # error, e_x, e_y = self.calculate_error(y, y_hat)
    # return K.mean(np.exp(e_x**2+e_y**2))
    # return K.prod((e_x**2+e_y**2))
    # return K.mean(K.sqrt(K.square(e_x)+K.square(e_y)))
    # return K.sum(K.sqrt(K.square(e_x)+K.square(e_y)))
    # return K.mean(y-y_hat, axis=1)
    # return K.exp(K.mean(K.sqrt(K.sum(K.square(y-y_hat), axis=0))))
    return K.mean(K.sqrt(K.sum(K.square(y-y_hat), axis=0)))

def base_model():
    model = Sequential()
    model.add(Dense(5, input_dim=1, init='normal', activation="sigmoid"))
    model.add(Dense(10, init='normal', activation="sigmoid"))
    model.add(Dense(10, init='normal', activation="sigmoid"))
    model.add(Dense(10, init='normal', activation="sigmoid"))
    model.add(Dense(10, init='normal', activation="sigmoid"))
    model.add(Dense(8, init='normal', activation="sigmoid"))
    model.add(Dense(8, init='normal', activation="sigmoid"))
    model.add(Dense(8, init='normal', activation="sigmoid"))
    model.add(Dense(6, init='normal', activation="sigmoid"))
    model.add(Dense(6, init='normal', activation="sigmoid"))
    model.add(Dense(6, init='normal', activation="sigmoid"))
    model.add(Dense(4, init='normal', activation="sigmoid"))
    model.add(Dense(4, init='normal', activation="sigmoid"))
    model.add(Dense(4, init='normal', activation="sigmoid"))
    model.add(Dense(3, init='normal', activation="sigmoid"))
    model.add(Dense(2, init='normal', activation="sigmoid"))

    model.compile(loss=localization_loss, optimizer='rmsprop')  # Using mse loss results in faster convergence
    return model

def main():
    fname_dataset = "ap_14_rssi.mat"
    folderLocation = os.path.dirname(os.path.realpath(__file__))
    # Create the dataset
    dataset = wifi_data.wifi_data(folder_location=folderLocation, filename=fname_dataset, normalize=False, missingValues=0, nTraining=2370, nTesting=750, nValidation=0, verbose=True)
    seed = 7
    np.random.seed(seed)
    clf = KerasRegressor(build_fn=base_model, nb_epoch=4000, batch_size=5000,verbose=0)

    clf.fit(dataset.train_set,dataset.pos_train)
    print clf.score(dataset.test_set, dataset.pos_test)
    print clf.predict(dataset.test_set)
    # res = clf.predict(X_test)
    # model.fit(dataset.train_set, dataset.pos_train, nb_epoch=5000, batch_size=5000, callbacks=callbacks)
    # pos_pred = model.predict(dataset.train_set)
    # # print pos_pred.shape, dataset.train_set
    # pos_pred = pos_pred.reshape(dataset.pos_train.shape)
    # print model.evaluate(dataset.train_set, dataset.pos_train)
    # print model.evaluate(dataset.test_set, dataset.pos_test)
    # shape = pos_pred.shape
    # for i in range(shape[0]):
    #     fig, ax = plt.subplots()
    #     point = dataset.pos_train[i,:]
    #     point_est = pos_pred[i,:]
    #     vec = np.vstack((point, point_est))
    #     print vec.shape
    #     # ax.plot(point[0], point[1],  'g*', label="Groundtruth")
    #     # ax.plot(point_est[0], point_est[1], 'b*', label="Approximated Function")
    #     ax.plot(vec[:,0], vec[:,1], 'r-')
    #     ax.grid()
    #     # ax.legend(loc='upper left', shadow=False)
    #
    #     plt.show()

if __name__ == "__main__":
    main()
