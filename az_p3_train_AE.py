import numpy as np
import matplotlib.pyplot as plt
from library.class_AE import get_compiled_model
from library.save_and_load_AE_model import save_my_model
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

anomalous_imgs_train = np.load(path_to_datasets + '/imgs_train_anomalous.npy')
normal_imgs_train = np.load(path_to_datasets + '/imgs_train_normal.npy')
imgs_test = np.load(path_to_datasets + '/imgs_test.npy')

# ============================================================================
""" System Paremeters"""
# SysParam = SystemParameters()
# # save the SysParam object as a pickle-type file
# with open(f'input/mySysParam.pickle', 'wb') as file:
#     pickle.dump(SysParam, file)

# load the pickle-type file
# cur_path = os.path.abspath(os.getcwd())
# folder_name = 'nRx10__B10dB_minus40p5deg__E5dB_10deg__K2__NLOS10'
# path_to_SysParam = os.path.join(cur_path, 'input/' + folder_name)

# with open(os.path.join(path_to_SysParam, 'mySysParam.pickle'), 'rb') as temp:
#     SysParam = pickle.load(temp)

###
num_angles = 180

###
angles = np.linspace(-num_angles/2, num_angles/2, num_angles)

# ============================================================================
plt.grid()
plt.plot(angles, anomalous_imgs_train[0])
plt.title("An Anomalous Spectrum")
plt.show()

""" Plot an normal SPECTRUM """
plt.grid()
plt.plot(angles, normal_imgs_train[0])
plt.title("A Normal Spectrum")
plt.show()

# =============================================================================
""" Compile and train the model """
model = get_compiled_model(num_angles)
history = model.fit(normal_imgs_train,
                    normal_imgs_train,
                    epochs=100, batch_size=2**6,
                    validation_data=(imgs_test, imgs_test),
                    shuffle=True, verbose=False)

# =============================================================================
""" Plot the loss function during training """
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

# =============================================================================
""" Save the trained AE model """
save_my_model(model, folder_name)
del model
