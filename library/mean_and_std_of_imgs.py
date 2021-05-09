import numpy as np


def mean_of_imgs(imgs):  # mean value
    n_imgs = len(imgs)
    num_angles = imgs.shape[1]
    avg_img = np.zeros_like(np.random.rand(num_angles))  # shape=(num_angles, )
    for i in range(n_imgs):
        avg_img += imgs[i]/n_imgs
    return avg_img


def std_of_imgs(imgs):  # standard deviation (std)
    avg_img = mean_of_imgs(imgs)
    std = np.power(avg_img - imgs, 2)
    std = np.mean(std, axis=0)  # shape = (num_angles, )
    std = np.power(std, 0.5)  # shape = (num_angles, )
    return std
