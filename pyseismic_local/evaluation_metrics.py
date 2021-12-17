import numpy as np
import math


def snr(signal_a, signal_b):
    signal_a = np.reshape(signal_a, (np.product(signal_a.shape), 1))
    signal_b = np.reshape(signal_b, (np.product(signal_b.shape), 1))
    if np.max(signal_a - signal_b) == 0 and np.min(signal_a - signal_b) == 0:
        snr_float = 99.9999
    else:
        snr_float = -20 * math.log10(np.linalg.norm(signal_a - signal_b) / np.linalg.norm(signal_a))
    return round(snr_float, 4)


#  We denote the set of pixels that belong to class i as Gi,
#  and the set of pixels classified as class i as Fi,
# then, the set of correctly classified pixels is Gi ∩ F i.
def get_classification_scores(Y_true, Y_pred, n_class):
    Y_true = Y_true.flatten()
    Y_pred = Y_pred.flatten()
    # print('Y_true.shape: ' + str(Y_true.shape))

    Y_true = np.array(Y_true, dtype=np.uint8)
    Y_pred = np.array(Y_pred, dtype=np.uint8)

    mask = (Y_true >= 1) & (Y_true <= n_class)
    # print('mask.shape: ' + str(mask.shape))
    # print('mask: ' + str(mask))

    hist = np.zeros((n_class, n_class))
    # print('hist.shape: ' + str(hist.shape))

    hist = hist + np.bincount(n_class * (Y_true[mask] - 1).astype(int) + (Y_pred[mask] - 1).astype(int),
                              minlength=n_class ** 2).reshape(n_class, n_class)

    # print(str(hist))
    #
    # print(str(hist.sum()))
    # print(str(hist.sum(axis=1)))  # sum over rows
    # accuracy_over_all_classes is the percentage of pixels over all classes
    accuracy_over_all_classes = np.diag(hist).sum() / hist.sum()

    # accuracy_for_a_class is the percentage of pixels that are correctly classified in a class i
    accuracy_for_a_class = np.diag(hist) / hist.sum(axis=1)
    # print(np.diag(hist))
    # print(hist.sum(axis=1))

    # We define the average_class_accuracy as the average of accuracy_for_a_class over all classes
    average_class_accuracy = np.nanmean(accuracy_for_a_class)

    # Intersection over union (IoUi) is defined as the number of elements of the intersection of Gi and Fi over the number of elements of their union set,
    # This metric measures the overlap between the two sets, and it should be one if and only if all pixels were correctly classified.
    IoU = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    # when we average IoU over all classes, we arrive at the average intersection over union (average_IoU)
    # Participants will be judged primarily using the average of the Intersection over Union (IoU) scores for the six facies categories,
    # although the Organizers reserve the right to use additional qualitative and quantitative metrics to gauge performance.
    average_IoU = np.nanmean(IoU)

    # To prevent this metric from being overly sensitive to small classes, it is common to weigh each class by its size.
    # The resulting metric is known as FWIU:
    class_weight = hist.sum(axis=1) / hist.sum()  # fraction of the pixels that come from each class
    # print(class_weight)
    weighted_IoU = (class_weight[class_weight > 0] * IoU[class_weight > 0]).sum()
    return accuracy_over_all_classes, accuracy_for_a_class, average_class_accuracy, IoU, average_IoU, weighted_IoU
