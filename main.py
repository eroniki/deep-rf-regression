#!/usr/bin/python

import numpy as np
import wifi_data
import os
import nn

def localization_loss(y, y_pred):
    error = y-y_pred
    e_x = error[:, 0]
    e_y = error[:, 1]
    return np.mean(e_x**2+e_y**2, axis=0)

def main():
    fname_dataset = "full_data_ucm.mat"
    fname_model = "model_nn.h5"
    folderLocation = os.path.dirname(os.path.realpath(__file__))
    dataset = wifi_data.wifi_data(folderLocation, fname_dataset, True, False, 2370, 750, 0)
    # dataset = wifi_data.wifi_data(folderLocation, fname_dataset, True, False, 2120, 500, 500)

    print "Training Shape: ", dataset.train_set.shape, dataset.pos_train.shape
    print "Test Shape: ", dataset.test_set.shape, dataset.pos_test.shape
    if dataset.valid_set != None:
        print "Validation Shape: ", dataset.valid_set.shape, dataset.pos_valid.shape

    myNN = nn.nn(model_name=fname_model, location=folderLocation, valid_split = 0.2)
    if not myNN.model_exists:
        myNN.fit_model(dataset.train_set, dataset.pos_train)

    # print "evaluation:"
    # print myNN.evaluate_model(dataset.valid_set, dataset.pos_valid)

    loc_hat = myNN.predict(dataset.test_set)
    mse = localization_loss(loc_hat, dataset.pos_test)

    print "\n", mse

    answer = raw_input("Would you like to save the model? Y/n\n")
    if answer[0] == 'y' or answer[0] == 'Y':
        myNN.save_model(folderLocation, fname_model)

if __name__ == "__main__":
    main()
