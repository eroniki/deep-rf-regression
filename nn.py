#!/usr/bin/python
import os
import time
import signal
import sys
import shutil

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adagrad, RMSprop, Adam
from keras import backend as K
from keras.regularizers import l2, activity_l2
from keras.callbacks import Callback, TensorBoard

import numpy as np
from matplotlib import pyplot as plt

class nn(object):
    """docstring for nn."""
    def __init__(self, model_name, location, valid_split, dropout=0.2, epoch=2000, batch_size=512):
    # def __init__(self, depth, dims, activations, epoch, model_name, location):
        super(nn, self).__init__()
        self.model = None
        self.full_file = None
        self.sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.ada = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
        self.adam = Adam()
        self.rmsprop = RMSprop(lr=0.01, clipnorm=1., decay=0.0001)
        self.optimizer = self.sgd
        self.model_name = model_name
        self.model_location = location
        self.model_exists = False
        self.answer = "n"
        self.valid_split = valid_split
        self.epoch = epoch
        self.batch_size = batch_size
        self.dropout = dropout
        # self.loss_function = loss_function
        self.regularizer = l2(0.01)
        self.regularizer = None
        self.log_dir = "./logs"

        self.callbacks = [
            TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=False)
        ]

        self.full_file = self.construct_full_location(self.model_location, self.model_name)

        signal.signal(signal.SIGINT, self.signal_handler)

        if self.check_model_exists(self.full_file):
            print "Model already exists!"
            self.answer = self.prompt_user("The model already exists, would you like to override? Y/n\n")
            if self.answer[0] == 'y' or self.answer[0] == 'Y':
                self.model_exists = False # Pretend model does NOT exist
                self.init_model()
                self.init_layers()
                self.compile_model()
            else:
                self.model_exists = True
                self.model = self.load_model(self.full_file)
        else:
            self.model_exists = False
            self.init_model()
            self.init_layers()
            self.compile_model()

    def init_model(self):
        self.model = Sequential()

    # TODO: Hard-coded layer structure should be relaxed!
    def init_layers(self):
        self.model.add(Dense(500, input_dim=132, init='he_normal', activation='relu', W_regularizer=self.regularizer, b_regularizer=self.regularizer))
        self.model.add(Dropout(0.5))
        for i in range(132, 6, -3):
            self.model.add(Dense(i, input_dim=132, init='he_normal', activation='relu', W_regularizer=self.regularizer, b_regularizer=self.regularizer))
            self.model.add(Dropout(0.5))


        self.model.add(Dense(5, init='he_normal', activation='relu', W_regularizer=self.regularizer, b_regularizer=self.regularizer))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4, init='he_normal', activation='relu', W_regularizer=self.regularizer, b_regularizer=self.regularizer))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3, init='he_normal', activation='relu', W_regularizer=self.regularizer, b_regularizer=self.regularizer))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, init='he_normal', activation='relu', W_regularizer=self.regularizer, b_regularizer=self.regularizer))

    def signal_handler(self, signal, frame):
        shutil.rmtree(self.log_dir, ignore_errors=False, onerror=None)
        # os.rmdir(self.log_dir, *, dir_fd=None)
        print "Clean-up done. Gracefully exited"
        sys.exit(0)

    def prompt_user(self, prompt):
        return raw_input(prompt)

    def plot_cdf(self, error, nbins):
        hist, bins = self.create_error_histogram(error, nbins)
        cdf = np.true_divide(np.cumsum(hist), np.sum(hist))
        plt.plot(bins[1:], cdf)
        plt.grid()
        plt.show()

    def create_error_histogram(self, error, nbins):
        hist, bins = np.histogram(error, nbins)
        return hist, bins

    def compile_model(self):
        # self.model.compile(loss=self.localization_loss, optimizer=self.optimizer)
        self.model.compile(loss=self.localization_loss, optimizer=self.optimizer)

    def fit_model(self, x_train, y_train):
        self.model.fit(x_train, y_train, nb_epoch=self.epoch, batch_size=self.batch_size, verbose=2, shuffle=True, validation_split = self.valid_split, callbacks=self.callbacks)

    def evaluate_model(self, x_valid, y_valid):
        return self.model.evaluate(x_valid, y_valid, batch_size=self.batch_size)

    def predict(self, x):
        return self.model.predict(x, batch_size=self.batch_size, verbose=1)

    def calculate_error(self, y, y_hat):
        error = y-y_hat
        e_x = error[:, 0]
        e_y = error[:, 1]
        return error, e_x, e_y

    def residual_analysis(self, y, y_hat, plot_cdf=True, nbins=10):
        error, e_x, e_y = self.calculate_error(y, y_hat)
        mse_x = np.mean(np.sqrt(e_x**2), axis=0)
        mse_y = np.mean(np.sqrt(e_y**2), axis=0)
        mse = np.mean(np.sqrt(e_x**2+e_y**2))

        if plot_cdf:
            self.plot_cdf(np.sqrt(e_x**2+e_y**2), nbins)

        return mse, mse_x, mse_y

    def localization_loss(self, y, y_hat):
        error, e_x, e_y = self.calculate_error(y, y_hat)
        # return K.mean(np.exp(e_x**2+e_y**2))
        return K.prod((e_x**2+e_y**2))

    def create_model(self, arg):
        pass

    def initialize_params(self, arg):
        pass

    def load_model(self, full_file):
        return load_model(full_file)

    def save_model(self, folder_location, fname):
        timestamp = int(time.time())
        fname = str(timestamp) + "_" + fname
        full_file = self.construct_full_location(folder_location, fname)
        self.model.save(full_file)

    def check_model_exists(self, full_file):
        try:
            exists = os.path.exists(full_file)
        except Exception as e:
            print "Exception caught!"
            print e
        return exists

    def construct_full_location(self, folder_location, fname):
        return folder_location + "/" + fname

def main():
    pass

if __name__ == "__main__":
    main()
