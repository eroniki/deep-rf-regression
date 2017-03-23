#!/usr/bin/python
from __future__ import division

import numpy as np
import numpy.matlib
import scipy as sp
import wifi_data
import os
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import mixture
from sklearn.manifold import TSNE
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input, merge, Merge, Conv1D
from keras.optimizers import SGD
from keras.utils.visualize_util import plot
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback, TensorBoard

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

def by_xy(x_train, y_train, x_test, y_test, log_dir):
    callback = [
        TensorBoard(log_dir=log_dir, histogram_freq=10, write_graph=True, write_images=False)
    ]
    inputs = Input(shape=(8,))

    x = Dense(10, init='uniform', activation="tanh")(inputs)
    x = Dropout(0.2)(x)
    x = Dense(8, init='uniform', activation="tanh")(x)
    x = merge([x, inputs], mode='sum')

    x = Dropout(0.1)(x)
    x = Dense(10, init='uniform', activation="tanh")(x)
    predictions = Dense(2, init='uniform', activation="tanh")(x)


    model = Model(input=inputs, output=predictions)
    plot(model, to_file='model_by_xy.png', show_shapes=True)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse',
                  optimizer=sgd,
                  metrics=['mean_squared_error', 'mean_squared_logarithmic_error'])

    model.fit(x_train, y_train,
              nb_epoch=500,
              batch_size=1000,
              verbose=2,
              shuffle=True,
              validation_split=0.25,
              callbacks=callback)

    # # print model.summary()
    score = model.evaluate(x_test, y_test, batch_size=50)
    return score

def by_number(x_train, y_train, x_test, y_test, log_dir):
    callback = [
        TensorBoard(log_dir=log_dir, histogram_freq=10, write_graph=False, write_images=False)
    ]
    inputs = Input(shape=(8,))
    x = Dense(15, init='uniform', activation="relu")(inputs)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(8, init='uniform', activation="relu")(x)
    x = merge([x, inputs], mode='sum')

    x = Dense(15, init='uniform', activation="relu")(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(8, init='uniform', activation="relu")(x)
    x = merge([x, inputs], mode='sum')

    x = Dense(15, init='uniform', activation="relu")(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(8, init='uniform', activation="relu")(x)
    x = merge([x, inputs], mode='sum')

    x = Dense(15, init='uniform', activation="relu")(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(8, init='uniform', activation="relu")(x)
    x = merge([x, inputs], mode='sum')
    x = Dense(15, init='uniform', activation="relu")(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    predictions = Dense(175, init='uniform', activation="softmax")(x)


    model = Model(input=inputs, output=predictions)
    plot(model, to_file='model_by_number.png', show_shapes=True)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy', "precision", "recall", "fbeta_score"])

    model.fit(x_train, y_train,
              nb_epoch=500,
              batch_size=1000,
              verbose=2,
              shuffle=True,
              validation_split=0.25,
              callbacks=callback)

    # # print model.summary()
    score = model.evaluate(x_test, y_test, batch_size=50)
    return score

def fit_gaussian(X):
    gmm = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(X)
    gmm.predict(X)
    return gmm.means_, gmm.covariances_

def naiveBayes(x_train, y_train, x_test, y_test):
    x_train = np.abs(x_train)
    x_test = np.abs(x_test)
    mnb  = MultinomialNB()
    mdl = mnb.fit(x_train, y_train)
    y_pred = mdl.predict(x_test)
    prob = mdl.predict_proba(x_test)
    return (y_test == y_pred).sum()/y_test.size, prob

def localization_error(y, y_hat):
    return np.sqrt(np.sum(np.power((y-y_hat),2), axis=1))

def cdf(error, nBins):
    countsError, binsError = np.histogram(error, nBins);
    f = np.cumsum(countsError) / np.sum(countsError);
    return binsError, countsError, f

def showTSNE(data, labels, nclass):
    colors = plt.cm.Set3(np.linspace(0, 1, nclass))
    # print type(colors), colors.shape
    # print colors[dataset.grid_numbers_train.ravel(),:].shape
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    t = model.fit_transform(data)
    print t.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(t[:,0], t[:,1], dataset.grid_numbers_train.ravel(), cmap=plt.get_cmap("magma"))
    cax = ax.scatter(t[:,0], t[:,1], labels(), color=colors[dataset.grid_numbers_train[0:300].ravel(),:])
    plt.show()

def main():
    fname_dataset = "hancock_data.mat"
    fname_model = "grid_classifier.h5"
    folderLocation = os.path.dirname(os.path.realpath(__file__))
    # Create the dataset
    dataset = wifi_data.wifi_data(folder_location=folderLocation, filename=fname_dataset, normalize=False, missingValues=0, nTraining=1200, nTesting=343, nValidation=0, verbose=False)
    # Show the dataset properties
    print "WiFi Set: ", dataset.wifi_train.shape, dataset.wifi_test.shape
    print "BT Set: ", dataset.bt_train.shape, dataset.bt_test.shape
    print "LoRa Set: ", dataset.lora_train.shape, dataset.lora_test.shape

    print "Grid Labels (by XY): ", dataset.grid_labels_train.shape, dataset.grid_labels_test.shape
    print "Grid Labels (by #): ", dataset.grid_numbers_train.shape, dataset.grid_numbers_test.shape

    # Validation split can be omitted, because Keras can create a validation
    # split from the training set
    if dataset.wifi_valid != None:
        print "Validation Shapes: \n"
        print "\tWiFi: ", dataset.wifi_valid.shape
        print "\tBT: ", dataset.bt_valid.shape
        print "\tLoRa: ", dataset.lora_valid.shape
        print "\tGrid (by xy-): ", dataset.grid_labels_valid.shape
        print "\tGrid (by #): ", dataset.grid_numbers_valid.shape


    grid_numbers_train_categorical = to_categorical(dataset.grid_numbers_train, 175)
    grid_numbers_test_categorical = to_categorical(dataset.grid_numbers_test, 175)

    # print np.amin(dataset.lora_train), np.amax(dataset.lora_train), np.count_nonzero(np.isnan(dataset.lora_train))
    # print np.amin(dataset.bt_train), np.amax(dataset.bt_train), np.count_nonzero(np.isnan(dataset.bt_train))
    # print np.amin(dataset.wifi_train), np.amax(dataset.wifi_train), np.count_nonzero(np.isnan(dataset.wifi_train))
    accxy, _ = naiveBayes(dataset.lora_train,dataset.grid_numbers_train.ravel(),dataset.lora_test, dataset.grid_numbers_test.ravel())
    accx, probx = naiveBayes(dataset.lora_train, dataset.grid_labels_train[:,0].ravel(), dataset.lora_test, dataset.grid_labels_test[:,0].ravel())
    accy, proby = naiveBayes(dataset.lora_train, dataset.grid_labels_train[:,1].ravel(), dataset.lora_test, dataset.grid_labels_test[:,1].ravel())
    print probx.shape, proby.shape
    # print np.min(dataset.grid_labels_test[:,0]), np.max(dataset.grid_labels_test[:,0])
    # print np.min(dataset.grid_labels_test[:,1]), np.max(dataset.grid_labels_test[:,1])

    # print np.array([np.argmax(probx, axis=1), np.argmax(proby, axis=1)]) == dataset.grid_labels_test
    sx = 0
    sy = 0
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    # fig.colorbar()
    for i in range(0,343):
        ax.clear()
        px = probx[i,:].reshape(1,probx[i,:].size)
        py = proby[i,:].reshape(proby[i,:].size,1)
        # print "px shape: ", px.shape, "py shape", py.shape
        belief_x = np.matlib.repmat(px, 25,1)
        belief_y = np.matlib.repmat(py, 1,7)
        # print belief_x.shape, belief_y.shape
        belief = np.multiply(belief_x, belief_y)
        belief = belief / np.sum(belief)
        im = ax.imshow(belief, extent=[0, 6, 24, 0], aspect='auto', interpolation='gaussian', origin='upper')
        print "gt: ", dataset.grid_labels_test[i,:], "x: ", np.argmax(px), "y: ", np.argmax(py)
        # ax.plot(dataset.grid_labels_test[i,0],dataset.grid_labels_test[i,1], 'k^')
        # fig.colorbar(im, cax=cax)
        # fig.canvas.draw()
        # fig.savefig(str(i).join(('outputs/', '.png')))

    print "overall: ", accxy, " x: ", accx, " y: ", accy


    e = localization_error(dataset.grid_labels_test, np.vstack((np.argmax(probx, axis=1), np.argmax(proby, axis=1))).transpose())
    bins, counts, f = cdf(e, 100)
    bins = bins[1:]
    min_id = np.argmin(np.abs(f - 0.9))
    print bins[min_id], f[min_id]
    plt.plot(bins, f)
    plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    plt.show()
    # print by_xy(dataset.lora_train, dataset.grid_labels_train, dataset.lora_test, dataset.grid_labels_test, "./logs")
    # print by_number(dataset.lora_train, grid_numbers_train_categorical, dataset.lora_test, grid_numbers_test_categorical, "./logs")


if __name__ == "__main__":
    main()
