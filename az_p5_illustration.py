import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score
from library.save_and_load_AE_model import load_my_model
from library.mean_and_std_of_imgs import mean_of_imgs, std_of_imgs
from library.performance_metrics import detection_metrics

from library.plot_original_and_decoded_imgs import plot_a_normal_img_and_its_reconstruction
from library.plot_original_and_decoded_imgs import plot_an_anomalous_img_and_its_reconstruction
from library.plot_mean_and_var_of_imgs import plot_mean_and_std_of_anomalous_imgs
from library.plot_mean_and_var_of_imgs import plot_mean_and_std_of_normal_imgs

# from pathlib import Path

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

###
cur_path = os.path.abspath(os.getcwd())
path_to_datasets = os.path.join(cur_path, 'input/' + folder_name)
path_to_results = os.path.join(cur_path, 'results/' + folder_name)

# ============================================================================
imgs_train_anomalous = np.load(path_to_datasets + '/imgs_train_anomalous.npy')
imgs_test_anomalous = np.load(path_to_datasets + '/imgs_test_anomalous.npy')
imgs_train_normal = np.load(path_to_datasets + '/imgs_train_normal.npy')
imgs_test_normal = np.load(path_to_datasets + '/imgs_test_normal.npy')
imgs_test = np.load(path_to_datasets + '/imgs_test.npy')

# ============================================================================
num_angles = 180
angles = np.linspace(-num_angles/2, num_angles/2, num_angles)

# ============================================================================
""" Plot the average imgs along with their associated stds """
plt.figure(1)
plot_mean_and_std_of_anomalous_imgs(imgs_train_anomalous, angles)
plot_mean_and_std_of_normal_imgs(imgs_train_normal, angles)
# plt.ylim(ymin=0, ymax=0.5)

""" Calculate the AUC for average curves """
# avg_anomalous_img = mean_of_imgs(imgs_train_anomalous)
# AUC_for_avg_anomalous_img = auc(angles, avg_anomalous_img)
# #
# avg_normal_img = mean_of_imgs(imgs_train_normal)
# AUC_for_avg_normal_img = auc(angles, avg_normal_img)
# #
# print("AUC for the average 'anomalous' curve: ", AUC_for_avg_anomalous_img)
# print("AUC for the average 'normal' curve: ", AUC_for_avg_normal_img)

# ============================================================================
""" Load the decoded imgs """
imgs_test_normal_decoded = np.load(path_to_results
                                   + '/imgs_test_normal_decoded.npy')
imgs_test_anomalous_decoded = np.load(path_to_results
                                      + '/imgs_test_anomalous_decoded.npy')

""" Plot a normal img, then plot its reconstruction """
plt.figure(2)
plot_a_normal_img_and_its_reconstruction(imgs_test_normal[0],
                                         imgs_test_normal_decoded[0],
                                         angles)
# plt.ylim(ymax=0.1)

""" Plot an anomalous img, then plot its reconstruction """
plt.figure(3)
plot_an_anomalous_img_and_its_reconstruction(imgs_test_anomalous[0],
                                             imgs_test_anomalous_decoded[0],
                                             angles)
# plt.ylim(ymax=0.1)

# ============================================================================
