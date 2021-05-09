import numpy as np
from library.mean_and_std_of_imgs import mean_of_imgs, std_of_imgs
import matplotlib.pyplot as plt
import pickle

'''
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use CPU

""" Define the path from C:> to the project """
cur_path = os.path.abspath(os.getcwd())  # path to library
path_to_project = Path(cur_path).parent
path_to_input = os.path.join(path_to_project, 'input')

###
anomalous_imgs_train = np.load(os.path.join(path_to_input,
                                            'imgs_train_anomalous.npy'))
anomalous_imgs_test = np.load(os.path.join(path_to_input,
                                           'imgs_test_anomalous.npy'))
normal_imgs_train = np.load(os.path.join(path_to_input,
                                         'imgs_train_normal.npy'))
normal_imgs_test = np.load(os.path.join(path_to_input,
                                        'imgs_test_normal.npy'))
imgs_test = np.load(os.path.join(path_to_input,
                                 'imgs_test.npy'))

# ============================================================================
""" System Paremeters"""
with open(
        os.path.join(path_to_input, 'mySysParam.pickle'),
        'rb') as temp:
    SysParam = pickle.load(temp)

num_angles = SysParam.num_angles
angles = np.linspace(-num_angles/2, num_angles/2, num_angles)
'''

"""calculate the average values"""
# avg_anomalous_img = mean_of_imgs(anomalous_imgs_train)
# avg_normal_img = mean_of_imgs(normal_imgs_train)

"""calculate the standard deviation (std)"""
# std_anomalous = std_of_imgs(anomalous_imgs_train)
# std_normal = std_of_imgs(normal_imgs_train)


def plot_mean_and_std_of_anomalous_imgs(imgs, angles):
    """calculate the average value"""
    avg_img = mean_of_imgs(imgs)
    """calculate the standard deviation (std)"""
    std = std_of_imgs(imgs)
    """ it's time to plot a figure"""
    opacity = 0.2
    plt.plot(angles, avg_img, label='(With Eve) Average of anomalous curves',
             linestyle='-', linewidth=2, color='r')
    plt.fill_between(angles,
                     avg_img - std,
                     avg_img + std, color='r', alpha=opacity,
                     label=r'(With Eve) Region of $(\mu-\sigma, \mu+\sigma)$')
    plt.legend(loc='best', fontsize=10)
    plt.xlabel(r'$\theta$', fontsize=12)
    plt.ylabel(r'$S(\theta)$', fontsize=12)
    plt.xlim(-90, 90)
    plt.tight_layout()
    return None


def plot_mean_and_std_of_normal_imgs(imgs, angles):
    """calculate the average value"""
    avg_img = mean_of_imgs(imgs)
    """calculate the standard deviation (std)"""
    std = std_of_imgs(imgs)
    """ it's time to plot a figure"""
    opacity = 0.2
    plt.plot(angles, avg_img, label='(Without Eve) Average of normal curves',
             linestyle='--', linewidth=2, color='k')
    plt.fill_between(angles,
                     avg_img - std,
                     avg_img + std, color='k', alpha=opacity,
                     label=r'(Without Eve) Region of $(\mu-\sigma, \mu+\sigma)$')
    plt.legend(loc='best', fontsize=10)
    plt.xlabel(r'$\theta$', fontsize=12)
    plt.ylabel(r'$S(\theta)$', fontsize=12)
    plt.xlim(-90, 90)
    plt.tight_layout()
    return None


# plt.figure(1)
# plot_mean_and_std_of_anomalous_imgs(anomalous_imgs_train, angles)
# plt.figure(1)
# plot_mean_and_std_of_normal_imgs(normal_imgs_train, angles)


# =============================================================================
"""Show a normal signal and an abnormal signal together"""
# plt.figure(1)
# plt.plot(angles, anomalous_imgs_train[0], label='Anomalous',
#           linestyle='-', linewidth=2, color='r')  # Plot a normal signal
# plt.plot(angles, normal_imgs_train[0], label='Normal',
#           linestyle='--', linewidth=2, color='k')  # Plot an anomalous signal
# plt.xlabel('Direction of Arrival', fontsize=12)
# plt.ylabel('Spectrum', fontsize=12)
# plt.legend(loc='best', fontsize=12)
# plt.show()

############
# plt.figure(2)
# for i in range(50):
#     if i == 0:
#         plt.plot(angles, anomalous_imgs_train[i], label='Abnormal',
#                   linestyle='-', linewidth=1, color='r')  # a normal signal
#         plt.plot(angles, normal_imgs_train[i], label='Normal',
#                   linestyle='--', linewidth=1, color='k')  # an anomalous signal
#     else:
#         plt.plot(angles, anomalous_imgs_train[i], 
#                   linestyle='-', linewidth=1, color='r')  # a normal signal
#         plt.plot(angles, normal_imgs_train[i],
#                   linestyle='--', linewidth=1, color='k')  # an anomalous signal
# plt.xlabel('Direction of Arrival', fontsize=12)
# plt.ylabel('Spectrum', fontsize=12)
# plt.legend(loc='best', fontsize=12)
# plt.show()

############