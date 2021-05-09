import numpy as np
from library.mean_and_std_of_imgs import mean_of_imgs


def compute_TP_and_FN(threshold, normal_imgs_train_decoded,
                      normal_imgs_test_decoded, actual_imgs_are_normal=True):
    TPos = 0  # True Positive = True 'NORMAL'
    FNega = 0  # False Negative = False 'ANOMALOUS'
    avg_normal_img = mean_of_imgs(normal_imgs_train_decoded)
    for i in range(len(normal_imgs_test_decoded)):
        an_unknown_img_decoded = normal_imgs_test_decoded[i]
        diff_from_normal_means = avg_normal_img - an_unknown_img_decoded
        SSD_from_normal_means = np.linalg.norm(diff_from_normal_means)**2  
        # print('SSD_from_normal_means =', SSD_from_normal_means)
        ###
        if SSD_from_normal_means <= threshold:
            # print("The unknown img is closer to 'NORMAL' imgs")
            if actual_imgs_are_normal == True:
                TPos += 1
            else:
                return None
        ###
        else:
            # print("The unknown img is closer to 'ANOMALOUS' imgs")
            if actual_imgs_are_normal == True:
                FNega += 1
            else:
                return None
        ###
    return TPos, FNega


def compute_FP_and_TN(threshold, normal_imgs_train_decoded,
                      anomalous_imgs_test_decoded, actual_imgs_are_anomalous=True):
    FPos = 0  # False Positive = False 'NORMAL'
    TNega = 0  # True Negative = True 'ANOMALOUS'
    avg_normal_img = mean_of_imgs(normal_imgs_train_decoded)
    for i in range(len(anomalous_imgs_test_decoded)):
        an_unknown_img_decoded = anomalous_imgs_test_decoded[i]
        diff_from_normal_means = avg_normal_img - an_unknown_img_decoded
        SSD_from_normal_means = np.linalg.norm(diff_from_normal_means)**2  
        # print('SSD_from_normal_means =', SSD_from_normal_means)
        ###
        if SSD_from_normal_means <= threshold:
            # print("The unknown img is closer to 'NORMAL' imgs")
            if actual_imgs_are_anomalous == True:
                FPos += 1
            else:
                return None
        ###
        else:
            # print("The unknown img is closer to 'ANOMALOUS' imgs")
            if actual_imgs_are_anomalous == True:
                TNega += 1
            else:
                return None
        ###
    return FPos, TNega


def detection_metrics(threshold, normal_imgs_train_decoded,
                      normal_imgs_test_decoded, anomalous_imgs_test_decoded):
    TPos, FNega = compute_TP_and_FN(threshold,
                                    normal_imgs_train_decoded,
                                    normal_imgs_test_decoded,
                                    actual_imgs_are_normal=True)
    FPos, TNega = compute_FP_and_TN(threshold,
                                    normal_imgs_train_decoded,
                                    anomalous_imgs_test_decoded,
                                    actual_imgs_are_anomalous=True)
    
    acc = (TPos + TNega)/(TPos + TNega + FPos + FNega)
    TPR = TPos/(TPos + FNega)  # True Positive Rate = Recall = Sensitivity
    FPR = FPos/(FPos + TNega)  # False Positive Rate = Fall-out
    FNR = FNega/(FNega + TPos)  # False Negative Rate = Miss rate
    TNR = TNega/(TNega + FPos)  # True Negative Rate = Specificity  
    return acc, TPR, FPR, FNR, TNR