import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

from class_SysParam import SystemParameters
from library.define_folder_name import name_of_folder
import pickle

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use CPU

""" Load the system parameters """
SysParam = SystemParameters()
n_Rx = SysParam.n_Rx  # number of antennas
snrdB_Bob = SysParam.snrdB_Bob  # in dB
snrdB_Eve = SysParam.snrdB_Eve  # in dB
DOA_Bob = SysParam.DOA_Bob  # in degrees
DOA_Eve = SysParam.DOA_Eve  # in degrees
K = SysParam.Rician_factor
NLOS = SysParam.n_NLOS_paths

""" Name the folder that contains the data """
folder_name = name_of_folder(n_Rx,
                             snrdB_Bob, DOA_Bob,
                             snrdB_Eve, DOA_Eve,
                             K, NLOS)

""" Save the system parameters """
cur_path = os.path.abspath(os.getcwd())
path_to_SysParam = os.path.join(cur_path, 'input/' + folder_name)
path_to_the_data = os.path.join(cur_path, 'input/' + folder_name)

with open(os.path.join(path_to_SysParam, 'mySysParam.pickle'), 'rb') as temp:
    SysParam = pickle.load(temp)

# load the dataset
dataframe = pd.read_csv(os.path.join(path_to_the_data, 'spectrum.csv'),
                        header=None)
dataframe.head()
raw_data = dataframe.values

# The last element contains the labels
y = raw_data[:, -1]

# The other data points are the electrocadriogram data
imgs = raw_data[:, 0:-1]

imgs_train, imgs_test, y_train, y_test = \
    train_test_split(imgs, y, test_size=0.2, random_state=21)

"""Normalize the data to `[0,1]`"""
min_val = tf.reduce_min(imgs_train)
max_val = tf.reduce_max(imgs_train)
imgs_train = (imgs_train - min_val) / (max_val - min_val)
imgs_test = (imgs_test - min_val) / (max_val - min_val)
imgs_train = tf.cast(imgs_train, tf.float32)
imgs_test = tf.cast(imgs_test, tf.float32)

"""We train the AE using only the normal images, which are labeled in
this dataset as `0`. Separate the normal imgs from the abnormal imgs."""

y_train = y_train.astype(bool)  # yields an array of True and False
y_test = y_test.astype(bool)  # yields an array of True and False

imgs_train_anomalous = imgs_train[y_train]  # gets elements associated with True
imgs_test_anomalous = imgs_test[y_test]  # gets elements associated with True

imgs_train_normal = imgs_train[~y_train]
imgs_test_normal = imgs_test[~y_test]


# def anomalous_and_normal_imgs(X, y_bool,
#                               anomalies_are_labelled_as_True = True):
#     """If anomalies_are_labelled_as_1 = True"""
#     if anomalies_are_labelled_as_True is True:
#         X_anomalous = X[y_bool]
#         X_normal = X[~y_bool]
#     ###
#     if anomalies_are_labelled_as_True is False:
#         X_normal = X[y_bool]
#         X_anomalous = X[~y_bool]
#     return X_anomalous, X_normal

""" Save the useful data """
path_to_datasets = os.path.join(cur_path, 'input/' + folder_name)

np.save(path_to_datasets + '/imgs_train_anomalous.npy', imgs_train_anomalous)
np.save(path_to_datasets + '/imgs_train_normal.npy', imgs_train_normal)
np.save(path_to_datasets + '/imgs_test.npy', imgs_test)
np.save(path_to_datasets + '/imgs_test_anomalous.npy', imgs_test_anomalous)
np.save(path_to_datasets + '/imgs_test_normal.npy', imgs_test_normal)
