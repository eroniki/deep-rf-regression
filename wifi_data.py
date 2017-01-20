#!/usr/bin/python
import scipy.stats as sis
import scipy.io as sio
import os
import time
import numpy as np

class wifi_data(object):
    """docstring for wifi_data."""
    def __init__(self, folder_location, filename, normalize, nTraining, nTesting, nValidation, verbose=False, missingValues=0):
        super(wifi_data, self).__init__()
        self.filename = filename
        self.folder_location= folder_location
        self.verbose = verbose
        self.normalize = normalize

        self.pos_train = None
        self.pos_test = None
        self.pos_valid = None
        self.test_set = None
        self.train_set = None
        self.valid_set = None

        self.__pos__ = None
        self.__feature_space__ = None
        self.__data_container__ = None
        self.__full_file_location__ = self.construct_full_location(self.folder_location, self.filename)
        self.__nSamples__ = None
        self.__nTraining__ = nTraining
        self.__nTesting__ = nTesting
        self.__nValid__ = nValidation
        self.__missingValues__ = missingValues

        if self.verbose:
            print "Folder Location: ", self.folder_location
            print "File Name: ", self.filename
            print "Full File Location: ", self.__full_file_location__


        if self.check_file_exists(self.__full_file_location__):
            if self.verbose:
                print "Database found!"
        else:
            if self.verbose:
                print "Database could NOT found!"
                return -1

        self.__data_container__ = self.read_file(self.__full_file_location__)

        self.__pos__, self.__feature_space__ = self.extract_data_from_container(self.__data_container__)

        if self.__missingValues__ != 0:
            self.__feature_space__[self.__feature_space__==0] = self.__missingValues__

        if verbose:
            print self.__pos__
            print self.__pos__.shape
            print self.__feature_space__
            print self.__feature_space__.shape

        if self.normalize:
            self.__feature_space__ = self.normalize_data(self.__feature_space__)

        self.train_set, self.pos_train, \
        self.test_set, self.pos_test,  \
        self.valid_set, self.pos_valid = self.split_data(self.__feature_space__, self.__pos__)

    def extract_data_from_container(self, container):
        try:
            pos = container["pos"]
            feature_space = container["featureSpace"]
        except Exception as e:
            print "Exception caught!"
            print e

        return pos, feature_space

    def split_data(self, fspace, pos):
        train_set = None
        pos_train = None
        test_set = None
        pos_test = None
        valid_set = None
        pos_valid = None

        self.__nSamples__ = self.__pos__.shape[0]

        [train_idx, test_idx, valid_idx] = self.get_randomly_sampled_indices(self.__nSamples__)

        train_set = fspace[train_idx,:]
        pos_train = pos[train_idx,:]

        test_set = fspace[test_idx,:]
        pos_test = pos[test_idx,:]

        if valid_idx!=None:
            valid_set = fspace[valid_idx,:]
            pos_valid = pos[valid_idx,:]

        return train_set, pos_train, test_set, pos_test, valid_set, pos_valid

    def get_randomly_sampled_indices(self, n_population):
        idx = np.arange(self.__nSamples__)

        train_idx = self.rand_sample(idx, self.__nTraining__)
        idx = self.update_population(idx, train_idx)

        test_idx = self.rand_sample(idx, self.__nTesting__)
        idx = self.update_population(idx, test_idx)

        valid_idx=None
        if self.__nValid__ !=0:
            valid_idx = self.rand_sample(idx, self.__nValid__)
            idx = self.update_population(idx, valid_idx)

        return train_idx, test_idx, valid_idx


    def update_population(self, population, individual):
        index = np.array([])

        for x in np.nditer(individual):
            index = np.argwhere(population==x)
            population = np.delete(population, index)
        return population

    def rand_sample(self, population, k):
        chosen = np.random.choice(population, k, False)
        return chosen

    def normalize_data(self, data):
        return sis.zscore(data)

    def read_file(self, arg):
        try:
            data_container = sio.loadmat(self.__full_file_location__)
        except Exception as e:
            print "Exception caught!"
            print e
        return data_container

    def check_file_exists(self, full_file):
        try:
            exists = os.path.exists(full_file)
        except Exception as e:
            print "Exception caught!"
            print e
        return exists

    def construct_full_location(self, folder_location, fname):
        return folder_location + "/" + fname

def main(arg):
    pass

if __name__ == "__main__":
    main()
