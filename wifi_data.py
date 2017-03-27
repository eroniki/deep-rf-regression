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

        self.wifi_train = None
        self.wifi_test = None
        self.wifi_valid = None

        self.bt_train = None
        self.bt_test = None
        self.bt_valid = None

        self.grid_labels_train = None
        self.grid_labels_test = None
        self.grid_labels_valid = None

        self.grid_numbers_train = None
        self.grid_numbers_test = None
        self.grid_numbers_valid = None

        self.grid_centers_train = None
        self.grid_centers_test =  None
        self.grid_centers_valid = None

        # self.pos_train = None
        # self.pos_train_classifier = None
        # self.pos_test = None
        # self.pos_test_classifier = None
        # self.pos_valid = None
        # self.pos_valid_classifier = None
        # self.test_set = None
        # self.train_set = None
        # self.valid_set = None

        self.__wifi_data__ = None
        self.__bt_data__ = None
        self.__lora_data__ = None
        self.__grid_labels_by_number__ = None
        self.__grid_labels_by_xy__ = None
        self.__grid_centers__ = None
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
            raise Exception("Database could NOT found!")

        self.__data_container__ = self.read_file(self.__full_file_location__)

        self.__grid_labels_by_xy__, self.__grid_labels_by_number__, \
        self.__wifi_data__, self.__bt_data__, self.__lora_data__, \
        self.__grid_centers__ = self.extract_data_from_container(self.__data_container__)

        if self.__missingValues__ != 0:
            self.__wifi_data__[self.__wifi_data__==0] = self.__missingValues__
            self.__bt_data__[self.__bt_data__==0] = self.__missingValues__
            self.__lora_data__[self.__lora_data__==0] = self.__missingValues__

        if verbose:
            print self.__grid_labels_by_xy__
            print self.__grid_labels_by_xy__.shape
            print self.__grid_labels_by_number__
            print self.__grid_labels_by_number__.shape
            print self.__wifi_data__
            print self.__wifi_data__.shape
            print self.__bt_data__
            print self.__bt_data__.shape
            print self.__lora_data__
            print self.__lora_data__.shape

        if self.normalize:
            self.__wifi_data__ = self.normalize_data(self.__wifi_data__)
            self.__bt_data__ = self.normalize_data(self.__bt_data__)
            self.__lora_data__ = self.normalize_data(self.__lora_data__)

        self.wifi_train, self.wifi_test, self.wifi_valid, \
        self.bt_train, self.bt_test, self.bt_valid, \
        self.lora_train, self.lora_test, self.lora_valid, \
        self.grid_labels_train, self.grid_labels_test, self.grid_labels_valid, \
        self.grid_numbers_train, self.grid_numbers_test, self.grid_numbers_valid,\
        self.grid_centers_train, self.grid_centers_test, self.grid_centers_valid = \
        self.split_data(self.__wifi_data__, self.__bt_data__, \
        self.__lora_data__, self.__grid_labels_by_xy__, self.__grid_labels_by_number__, self.__grid_centers__)

    def extract_data_from_container(self, container):
        try:
            grid_labels = container["grid_labels"]
            grid_labels_linear = container["grid_labels_linear"]
            data_wifi = container["data_wifi"]
            data_bt = container["data_bt"]
            data_lora = container["data_lora"]
            grid_centers = container["grid_centers"]
        except Exception as e:
            print "Exception caught!"
            print e

        return grid_labels, grid_labels_linear, data_wifi, data_bt, data_lora, grid_centers

    def split_data(self, wifi_data, bt_data, lora_data, grid_labels_by_xy, grid_labels_by_number, grid_centers):
        wifi_data_train = None
        wifi_data_test = None
        wifi_data_valid = None

        bt_data_train = None
        bt_data_test = None
        bt_data_valid = None

        lora_data_train = None
        lora_data_test = None
        lora_data_valid = None

        grid_labels_xy_train = None
        grid_labels_xy_test = None
        grid_labels_xy_valid = None

        grid_labels_by_number_train = None
        grid_labels_by_number_test = None
        grid_labels_by_number_valid = None

        grid_centers_train = None
        grid_centers_test = None
        grid_centers_valid = None

        print grid_centers.shape

        self.__nSamples__ = grid_labels_by_xy.shape[0]

        [train_idx, test_idx, valid_idx] = self.get_randomly_sampled_indices(self.__nSamples__)

        wifi_data_train = wifi_data[train_idx,:]
        bt_data_train = bt_data[train_idx,:]
        lora_data_train = lora_data[train_idx,:]

        grid_labels_xy_train = grid_labels_by_xy[train_idx,:]
        grid_labels_by_number_train = grid_labels_by_number[train_idx,:]
        grid_centers_train = grid_centers[train_idx, :]

        wifi_data_test = wifi_data[test_idx,:]
        bt_data_test = bt_data[test_idx,:]
        lora_data_test = lora_data[test_idx,:]

        grid_labels_xy_test = grid_labels_by_xy[test_idx,:]
        grid_labels_by_number_test = grid_labels_by_number[test_idx,:]
        grid_centers_test = grid_centers[test_idx, :]

        if valid_idx!=None:
            wifi_data_valid = wifi_data[valid_idx,:]
            bt_data_valid = bt_data[valid_idx,:]
            lora_data_valid = lora_data[valid_idx,:]

            grid_labels_xy_valid = grid_labels_by[valid_idx,:]
            grid_labels_by_number_valid = grid_labels_by_number[valid_idx,:]
            grid_centers_valid = grid_centers[valid_idx, :]


        return wifi_data_train, wifi_data_test, wifi_data_valid, \
        bt_data_train, bt_data_test, bt_data_valid, \
        lora_data_train, lora_data_test, lora_data_valid, \
        grid_labels_xy_train, grid_labels_xy_test, grid_labels_xy_valid, \
        grid_labels_by_number_train, grid_labels_by_number_test,\
        grid_labels_by_number_valid, grid_centers_train, grid_centers_test, \
        grid_centers_valid

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
