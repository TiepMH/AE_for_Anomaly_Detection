from library.generate_spectrums_and_save_them import generate_spectrums
from library.generate_spectrums_and_save_them import merge_two_datasets_into_one

from class_SysParam import SystemParameters
from library.define_folder_name import name_of_folder
import pickle

import os
cur_path = os.path.abspath(os.getcwd())

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


""" No_Attack = True    >>>    There is no attack from any eavesdropper

    No_Attack = False   >>>    An eavesdropper is attacking the network """


# Firstly, we generate spectrums in the case of non-eavesdropping attack
# The data is stored in 'input/MUSIC_spectrums_label_0.csv'
No_Attack = True
generate_spectrums(No_Attack, folder_name)

# Secondly, we generate spectrums in the case of an eavesdropping attack
# The data is stored in 'input/MUSIC_spectrums_label_1.csv'
No_Attack = False
generate_spectrums(No_Attack, folder_name)

# =============================================================================
""" It's time to merge 2 csv files into 1 csv file

link 1: https://stackoverflow.com/questions/2512386/how-to-merge-200-csv-files-in-python

"""
# the csv files are MUSIC_spectrums_label_0.csv and MUSIC_spectrums_label_1.csv
# the merged csv file is spectrum.csv

merge_two_datasets_into_one(folder_name)

# =============================================================================
