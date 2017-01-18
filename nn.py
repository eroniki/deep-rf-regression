#!/usr/bin/python
import os
import time
import json

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adagrad

from keras import backend as K

import numpy as np

class nn(object):
    """docstring for nn."""
    def __init__(self, model_name, location, valid_split):
    # def __init__(self, depth, dims, activations, epoch, model_name, location):
        super(nn, self).__init__()
        self.model = None
        self.optimizer = None
        self.full_file = None
        self.sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.ada = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
        self.model_name = model_name
        self.model_location = location
        self.model_exists = False
        self.answer = "n"
        self.valid_split = valid_split
        # self.loss_function = loss_function

        self.full_file = self.construct_full_location(self.model_location, self.model_name)
        print self.full_file, self.check_model_exists(self.full_file)
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

    def init_model(self):
        self.model = Sequential()

    def init_layers(self):
        self.model.add(Dense(256, input_dim=132, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, input_dim=132, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, input_dim=132, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(16, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(8, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, init='uniform'))
        self.model.add(Activation('relu'))

    def prompt_user(self, prompt):
        return raw_input(prompt)


    def compile_model(self):
        self.model.compile(loss=self.localization_loss,
        # self.model.compile(loss='mean_squared_error',
                      optimizer=self.sgd)

    def fit_model(self, x_train, y_train):
        self.model.fit(x_train, y_train, nb_epoch=200, batch_size=500, verbose=2, shuffle=False, validation_split = self.valid_split)

    def evaluate_model(self, x_valid, y_valid):
        return self.model.evaluate(x_valid, y_valid, batch_size=500)

    def predict(self, x):
        return self.model.predict(x, batch_size=500, verbose=1)

    def localization_loss(self, y, y_pred):
        error = y-y_pred
        e_x = error[:, 0]
        e_y = error[:, 1]
        return K.mean(e_x**2+e_y**2)

    def create_model(self, arg):
        pass

    def initialize_params(self, arg):
        pass

    def load_model(self, full_file):
        return load_model(full_file)

    def save_model(self, folder_location, fname):
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

# DNNRegressor
if __name__ == "__main__":
    main()
