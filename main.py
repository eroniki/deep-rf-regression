#!/usr/bin/python

import tensorflow as tf
import numpy as np

import wifi_data
import os
import nn


def main():
    folderLocation = os.path.dirname(os.path.realpath(__file__))
    fname = "full_data_ucm.mat"
    dataHandler = wifi_data.wifi_data(folderLocation, fname, True, False)

if __name__ == "__main__":
    main()
