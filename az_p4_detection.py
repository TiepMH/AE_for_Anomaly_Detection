import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score
from library.save_and_load_AE_model import load_my_model
from library.mean_and_std_of_imgs import mean_of_imgs, std_of_imgs
from library.performance_metrics import detection_metrics

from library.plot_original_and_decoded_imgs import plot_a_normal_img_and_its_reconstruction
from library.plot_original_and_decoded_imgs import plot_an_anomalous_img_and_its_reconstruction

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
""" Load the trained AE model """
model = load_my_model(num_angles, folder_name)

""" Pass normal imgs to get their decoded imgs """
imgs_test_normal_encoded = model.encoder(imgs_test_normal).numpy()
imgs_test_normal_decoded = model.decoder(imgs_test_normal_encoded).numpy()
# save imgs_test_normal_decoded
np.save(path_to_results + '/imgs_test_normal_decoded.npy',
        imgs_test_normal_decoded)
#
# plt.figure(2)
# plot_a_normal_img_and_its_reconstruction(imgs_test_normal[0],
#                                           imgs_test_normal_decoded[0],
#                                           angles)
# plt.ylim(ymax=0.1)

""" Pass anomalous imgs to get their decoded imgs """
imgs_test_anomalous_encoded = model.encoder(imgs_test_anomalous).numpy()
imgs_test_anomalous_decoded = model.decoder(imgs_test_anomalous_encoded).numpy()
# save imgs_test_anomalous_decoded
np.save(path_to_results + '/imgs_test_anomalous_decoded.npy',
        imgs_test_anomalous_decoded)
#
# plt.figure(3)
# plot_an_anomalous_img_and_its_reconstruction(imgs_test_anomalous[0],
#                                               imgs_test_anomalous_decoded[0],
#                                               angles)
# plt.ylim(ymax=0.1)

# ============================================================================
"""Detect anomalies based on SSD = sum_of_squared_differences"""
imgs_train_normal_encoded = model.encoder(imgs_train_normal).numpy()
imgs_train_normal_decoded = model.decoder(imgs_train_normal_encoded).numpy()

###
std = std_of_imgs(imgs_train_normal_decoded)
threshold = np.linalg.norm(std)**2

###
acc, TPR, FPR, FNR, TNR = detection_metrics(threshold,
                                            imgs_train_normal_decoded,
                                            imgs_test_normal_decoded,
                                            imgs_test_anomalous_decoded)

""" Show the detection performance """
print("accuracy =", acc)
print("sensitivity =", TPR)
print("specificity =", TNR)

""" Save the results """
results = {'TPR': TPR, 'FPR': FPR, 'FNR': FNR, 'TNR': TNR,
           'accuracy': acc,
           'sensitivity': TPR, 'TPR': TPR,
           'specificity': TNR, 'TNR': TNR,
           }


cur_path = os.path.abspath(os.getcwd())
with open(os.path.join(cur_path, 'results/' + folder_name + '/results.pickle'), 'wb') as temp:
    pickle.dump(results, temp)
