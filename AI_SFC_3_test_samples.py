from pyseismic_local.AI_ML_DL_functions import *
from pyseismic_local.AI_SeismicFaciesClassification import *
from pyseismic_local.evaluation_metrics import *

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import random
import segyio

np.random.seed(seed=0)  # for reproducibility
random.seed(0)



alphaX = 1.0
alphaY = 0.5

patch_rows = 992
patch_cols = 576

Y_channels = 1
Y_1 = 0
n_class = 6

model_folder_id = 'loss5'

if model_folder_id == 'loss1234':
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss1_lr100_pool1_kernelSize11_Net5_kernels16_repeat1'
    # model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss2_lr100_pool1_kernelSize11_Net5_kernels16_repeat1'
    # model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss3_lr100_pool1_kernelSize11_Net5_kernels16_repeat1'
    # model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss4_lr100_pool1_kernelSize11_Net5_kernels16_repeat1'

    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss1_lr200_pool1_kernelSize11_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss2_lr200_pool1_kernelSize11_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss3_lr200_pool1_kernelSize11_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss4_lr200_pool1_kernelSize11_Net5_kernels16_repeat1'

    disk_ID_model = 'D'
    stride = [45, 5]
    disk_ID_data = 'D'
    Y_1 = 0
if model_folder_id == 'loss5':
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr50_pool1_kernelSize11_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool2_kernelSize11_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat0'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr10_pool1_kernelSize11_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr1_pool1_kernelSize11_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr200_pool1_kernelSize11_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr500_pool1_kernelSize11_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat2'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize9_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize7_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize5_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize3_Net5_kernels16_repeat1'

    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr200_pool1_kernelSize9_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr200_pool1_kernelSize7_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr200_pool1_kernelSize5_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr200_pool1_kernelSize3_Net5_kernels16_repeat1'

    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr200_pool1_kernelSize11_Net5_kernels16_repeat0'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr200_pool1_kernelSize11_Net5_kernels16_repeat1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr200_pool1_kernelSize11_Net5_kernels16_repeat2'

    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr200_pool2_kernelSize11_Net5_kernels16_repeat1'

    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_actHide1_kernelSize11_Net5_kernels16_drop0_repeat1'
    # model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1_1'

    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize7_Net5_kernels16_repeat1_dila2_repro1'
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1_dila1_repro1_norm1'
    X_normal = 1.0
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1_dila1_repro1_norm43'
    X_normal = 5522.086 * (1.0 / 127.0)
    model_folder = 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1_dila1_repro1_norm5522'
    X_normal = 5522.086
    model_folder = '2patchR992C576_batch16_number10000_stride8_8_loss5_lr1000_pool1_kernelSize11_Net5_kernels16_repeat1_repro2'
    model_folder = '2patchR992C576_batch16_number10000_stride8_8_loss5_lr1000_pool1_kernelSize11_Net5_kernels16_repeat1_repro0'
    model_folder = 'patchR0992C0576_batch016_number10000_stride8_8_lr1000_kSize11_Res0_Net5_Crep2_repro0_kernels16_32_64_128_256_512'

    X_normal = 5522.086
    disk_ID_model = 'G'
    stride = [8, 8]

    disk_ID_data = 'G'
    Y_1 = 1

print(disk_ID_model)

module_name = 'AI_SeismicFaciesClassification'
module_data_dir = disk_ID_data + ':/BigData/' + module_name
module_model_dir = disk_ID_model + ':/BigData/' + module_name
training_data_dir = module_data_dir + '/' + 'training_data_' + str(int(patch_rows)) + 'x' + str(int(patch_cols)) + '_stride' + str(stride[0]) + '_' + str(stride[1]) + '_Ychannels' + str(Y_channels)
models_dir = module_model_dir + '/' + 'models'

figures_dir = './' + 'figures'
if not os.path.exists(figures_dir):
    os.mkdir(figures_dir)

figures_save = 1
paper_results_save_dir = './paper_results'
if not os.path.exists(paper_results_save_dir):
    os.mkdir(paper_results_save_dir)

model_log_dir = models_dir + '/' + model_folder

model_number = findLastCheckpoint(model_log_dir=model_log_dir)
# model_number = 10
model_id = model_log_dir + '/model_' + format(model_number, '03d') + '.hdf5'
log_csv_id = model_log_dir + '/log.csv'
fname = figures_dir + '/' + model_folder + '.jpg'
plot_training_validation_logs(log_csv_id)
if figures_save:
    plt.savefig(fname=fname, dpi=100, facecolor='w', edgecolor='w')

print(model_id)

model = load_model(model_id, compile=False)

X_channels = 1
X_max = 5522.086
X_min = -5195.5234

zoom_in = 0.1
test_data_number = 5
eval_scores = np.zeros(shape=(test_data_number, 5), dtype=np.float32)

for i in range(test_data_number):
    test_data = i
    X_id, Y_id, dataset_name = test_data_generator2D(test_data, training_data_dir)
    data_name_prefix = str(test_data) + '_' + dataset_name + '_' + model_folder[len(
        'patchR992C576_batch19_'):] + '_model' + format(model_number, '03d')
    ################################################################################
    if test_data == 0:
        testID = 1 - 1
        X_test = bin_data_reader2D_float32(X_id, patch_rows, patch_cols, X_channels, 1)
        Y_true = bin_data_reader2D_float32(Y_id, patch_rows, patch_cols, Y_channels, 1)
    else:
        if test_data != 3 and test_data != 4:
            X_test = segyio.tools.cube(X_id)
        else:
            X_test = bin_data_reader3D_float32(X_id, 1006, 1116, 841, X_channels, 1)
            X_test = np.reshape(X_test, (1006, 1116, 841))
        if test_data == 1 or test_data == 2:
            Y_true = segyio.tools.cube(Y_id)
        else:
            Y_true = np.ones(shape=X_test.shape, dtype='float32')

        # print('X_test.shape:' + str(X_test.shape))
        # print('Y_true.shape:' + str(Y_true.shape))
        if test_data != 3 and test_data != 4:
            X_test = X_test.transpose((2, 1, 0))
            Y_true = Y_true.transpose((2, 1, 0))
            # print('X_test.shape:' + str(X_test.shape))
            # print('Y_true.shape:' + str(Y_true.shape))
        if test_data == 1:
            testID = 1 - 1
            # 1006x782x590
            X_test = X_test[0:(X_test.shape[0] // 32) * 32, 0:(X_test.shape[1] // 32) * 32, -1]
            Y_true = Y_true[0:(Y_true.shape[0] // 32) * 32, 0:(Y_true.shape[1] // 32) * 32, -1]
        if test_data == 2:
            testID = 1 - 1
            # 1006x782x590
            X_test = X_test[0:(X_test.shape[0] // 32) * 32, -1, 0:(X_test.shape[2] // 32) * 32]
            Y_true = Y_true[0:(Y_true.shape[0] // 32) * 32, -1, 0:(Y_true.shape[2] // 32) * 32]
        if test_data == 3 or test_data == 5 or test_data == 7:
            if test_data == 3:
                testID = 590 - 1
            else:
                testID = random.randint(1, X_test.shape[2]) - 1
            X_test = X_test[0:(X_test.shape[0] // 32) * 32, 0:(X_test.shape[1] // 32) * 32, testID]
            Y_true = Y_true[0:(Y_true.shape[0] // 32) * 32, 0:(Y_true.shape[1] // 32) * 32, testID]
        if test_data == 4 or test_data == 6 or test_data == 8:
            if test_data == 4:
                testID = 782 - 1
            else:
                testID = random.randint(1, X_test.shape[1]) - 1
            X_test = X_test[0:(X_test.shape[0] // 32) * 32, testID, 0:(X_test.shape[2] // 32) * 32]
            Y_true = Y_true[0:(Y_true.shape[0] // 32) * 32, testID, 0:(Y_true.shape[2] // 32) * 32]

        # print('X_test.shape:' + str(X_test.shape))
        # print('Y_true.shape:' + str(Y_true.shape))
        X_test = np.reshape(X_test, (1, X_test.shape[0], X_test.shape[1], 1))
        Y_true = np.reshape(Y_true, (1, Y_true.shape[0], Y_true.shape[1], 1))
    X_test = X_test / X_normal
    # print('X_test min: ' + str(np.min(X_test)) + '   max:' + str(np.max(X_test)))
    # print('Y_true min: ' + str(np.min(Y_true)) + '   max:' + str(np.max(Y_true)))
    ################################################################################
    start_time = time.time()
    Y_pred = model.predict(X_test)
    if Y_1 == 1:
        Y_pred = Ypred_621(Y_pred)
    else:
        Y_pred = ReClassification(Y_pred)
    elapsed_time = time.time() - start_time
    # print('Y_pred min: ' + str(np.min(Y_pred[:])) + '   max:' + str(
    #     np.max(Y_pred[:])))
    if i == 3 or i == 4:
        Y_true_available = segyio.tools.cube(
            'D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' + 'TrainingData_Labels.segy')
        Y_true_available = Y_true_available.transpose((2, 1, 0))
        if i == 3:
            Y_true_available = Y_true_available[0:(Y_true_available.shape[0] // 32) * 32, 1:Y_true_available.shape[1],
                               testID]
        else:
            Y_true_available = Y_true_available[0:(Y_true_available.shape[0] // 32) * 32, testID,
                               0:Y_true_available.shape[2]]
        Y_true_available = np.reshape(Y_true_available, (1, Y_true_available.shape[0], Y_true_available.shape[1], 1))
        # print(Y_true_available.shape)
        # print(Y_true.shape)
        Y_true[:, :, 0:Y_true_available.shape[2], :] = Y_true_available
        if i == 3:
            eval_scores[i, 0] = snr(Y_true_available, Y_pred[:, :, 1:1 + Y_true_available.shape[2], :])
            accuracy_over_all_classes, accuracy_for_a_class, average_class_accuracy, IoU, average_IoU, weighted_IoU = get_classification_scores(
                Y_true_available, Y_pred[:, :, 1:1 + Y_true_available.shape[2], :], n_class)
        else:
            eval_scores[i, 0] = snr(Y_true_available, Y_pred[:, :, 0:Y_true_available.shape[2], :])
            accuracy_over_all_classes, accuracy_for_a_class, average_class_accuracy, IoU, average_IoU, weighted_IoU = get_classification_scores(
                Y_true_available, Y_pred[:, :, 0:Y_true_available.shape[2], :], n_class)
        eval_scores[i, 1] = accuracy_over_all_classes
        eval_scores[i, 2] = average_class_accuracy
        eval_scores[i, 3] = average_IoU
        eval_scores[i, 4] = weighted_IoU
        del Y_true_available
    else:
        if i == 1:
            eval_scores[i, 0] = snr(Y_true[:, :, 1:, :], Y_pred[:, :, 1:, :])
            accuracy_over_all_classes, accuracy_for_a_class, average_class_accuracy, IoU, average_IoU, weighted_IoU = get_classification_scores(
                Y_true[:, :, 1:, :], Y_pred[:, :, 1:, :], n_class)
        else:
            eval_scores[i, 0] = snr(Y_true, Y_pred)
            accuracy_over_all_classes, accuracy_for_a_class, average_class_accuracy, IoU, average_IoU, weighted_IoU = get_classification_scores(
                Y_true, Y_pred, n_class)
        eval_scores[i, 1] = accuracy_over_all_classes
        eval_scores[i, 2] = average_class_accuracy
        eval_scores[i, 3] = average_IoU
        eval_scores[i, 4] = weighted_IoU
    print('time=%4.3fs' % (elapsed_time))
    common_str = data_name_prefix + '_ID' + str(testID + 1) + '_' + str(X_test.shape[1]) + 'x' + str(
        X_test.shape[2]) + '_SNR= %2.4f' % (eval_scores[i, 0]) + '_MIoU= %2.4f' % (eval_scores[i, 3])
    print(common_str)

    ################################################################################
    fname = figures_dir + '/' + data_name_prefix + '_ID' + str(testID + 1) + '_' + str(X_test.shape[1]) + 'x' + str(
        X_test.shape[2]) + '_SNR= %2.4f' % (eval_scores[i, 0]) + '_MIoU= %2.4f' % (
                eval_scores[i, 3]) + '.jpg'
    plot_SFC_2D_results(X_test, Y_true, Y_pred, common_str, alphaX=alphaX, alphaY=alphaY)
    if 2 < i < 5 and figures_save:
        plt.savefig(fname=fname, dpi=100, facecolor='w', edgecolor='w')

    del test_data, X_id, Y_id, dataset_name, data_name_prefix, X_test, Y_true, Y_pred, testID, fname

plt.show()
