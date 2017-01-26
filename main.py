#!/usr/bin/python

import numpy as np
import wifi_data
import os
import nn
import gru

def main():
    fname_dataset = "full_data_ucm.mat"
    fname_model = "1484752125_model_nn.h5"
    fname_model_rnn = "1484752125_model_rnn.h5"
    folderLocation = os.path.dirname(os.path.realpath(__file__))
    # Create the dataset
    dataset = wifi_data.wifi_data(folder_location=folderLocation, filename=fname_dataset, normalize=False, missingValues=0, nTraining=2370, nTesting=750, nValidation=0, verbose=False)
    # Show the dataset properties
    print "Training Shape: ", dataset.train_set.shape, dataset.pos_train.shape
    print "Test Shape: ", dataset.test_set.shape, dataset.pos_test.shape
    # Validation split can be omitted, because Keras can create a validation
    # split from the training set
    if dataset.valid_set != None:
        print "Validation Shape: ", dataset.valid_set.shape, dataset.pos_valid.shape

    # Initiate the Neural Network Object
    myNN = nn.nn(model_name=fname_model, location=folderLocation, valid_split = 0.15, epoch=150)
    # myGRU = gru.gru(model_name=fname_model_rnn, location=folderLocation, valid_split = 0.2, epoch=100)

    # Check if the model already exists, which prevents re-training
    # If the model does NOT exist, train the model with the training samples
    # Otherwise, the already existing model will be used.
    if not myNN.model_exists:
        myNN.fit_model(dataset.train_set, dataset.pos_train)
        # myGRU.fit_model(dataset.train_set, dataset.pos_train)
    # Testing
    loc_hat = myNN.predict(dataset.test_set)
    # loc_hat = myGRU.predict(dataset.test_set)

    # Evaluation
    mse, mse_x, mse_y = myNN.residual_analysis(y=dataset.pos_test, y_hat=loc_hat, plot_cdf=True, nbins=100)
    # mse, mse_x, mse_y = myGRU.residual_analysis(y=dataset.pos_test, y_hat=loc_hat, plot_cdf=True, nbins=100)

    print "\nMSE: ", mse, "MSE_x: ", mse_x, " MSE_y: ", mse_y

    # answer = raw_input("Would you like to save the model? Y/n\n")
    # if answer[0] == 'y' or answer[0] == 'Y':
    #     myNN.save_model(folderLocation, fname_model)

if __name__ == "__main__":
    main()
