from pyseismic_local.AI_SeismicFaciesClassification import *
from pyseismic_local.evaluation_metrics import *

import os

import time
import numpy as np
from keras.models import load_model
import random
import segyio
import argparse

np.random.seed(seed=0)  # for reproducibility
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--gpuID', default=0, type=int, help='gpuID')
parser.add_argument('--model_folder_id', nargs='?', type=str, default='lr100', help='model_folder_id')
parser.add_argument('--models_disk', nargs='?', type=str, default='G', help='models_disk')
parser.add_argument('--iline_way', default=1, type=int, help='iline_way')
parser.add_argument('--Y_1', default=0, type=int, help='Y_1')
parser.add_argument('--model_number_begin', default=1, type=int, help='model_number_begin')
parser.add_argument('--model_number_end', default=1, type=int, help='model_number_end')
parser.add_argument('--plot_show', action='store_true', help='plot_show')
parser.add_argument('--X_normal', default=5522.086, type=float, help='X_normal')
args = parser.parse_args()

if args.gpuID == -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('CPU is used.')
elif args.gpuID == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('GPU device ' + str(args.gpuID) + ' is used.')
elif args.gpuID == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('GPU device ' + str(args.gpuID) + ' is used.')

X_normal = args.X_normal
print(X_normal)

Y_channels = 1
n_class = 6
figures_save = 1

module_name = 'AI_SeismicFaciesClassification'
disk_ID_model = args.models_disk
module_model_dir = disk_ID_model + ':/BigData/' + module_name
models_dir = module_model_dir + '/' + 'models'
models_path = models_dir + '/'

figures_dir = './' + 'figures'
if not os.path.exists(figures_dir):
    os.mkdir(figures_dir)

paper_results_save_dir = './paper_results'
if not os.path.exists(paper_results_save_dir):
    os.mkdir(paper_results_save_dir)

paper_results_big_dir = module_model_dir + '/' + 'results'
if not os.path.exists(paper_results_big_dir):
    os.mkdir(paper_results_big_dir)

model_number_check = list(range(args.model_number_begin, args.model_number_end + 1, 1))

X_channels = 1

zoom_in = 0.1
test_data = 3
X_id = 'D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/Training_Test1_Test2_Image_1006x1116x841.bin'
Y_id = ''
if args.iline_way == 1:
    dataset_name = 'All_iline'
else:
    dataset_name = 'All_xline'

data_name_prefix = str(test_data) + '_' + dataset_name + '_' + args.model_folder_id
print(data_name_prefix)

nt_all = 1006

xlines_all = 1116
ilines_all = 841

X_test = bin_data_reader3D_float32(X_id, nt_all, xlines_all, ilines_all, X_channels, 1)
X_test = np.reshape(X_test, (nt_all, xlines_all, ilines_all))
if args.iline_way != 1:
    X_test = X_test.transpose((0, 2, 1))
    xlines_all = X_test.shape[1]
    ilines_all = X_test.shape[2]

X_test = X_test / X_normal
print('X_test.shape:' + str(X_test.shape))

Y_true_available = segyio.tools.cube('D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' + 'TrainingData_Labels.segy')
Y_true_available = Y_true_available.transpose((2, 1, 0))
print(Y_true_available.shape)
xlines_true = Y_true_available.shape[1]
ilines_true = Y_true_available.shape[2]

# X_test.shape [1006 1116 841]
# time 992,  xline 1088
xline_pro = (xlines_all // 32) * 32
print(xline_pro)
nt_pro = (nt_all // 32) * 32
print(nt_pro)

X_test_blocks = np.zeros(shape=(nt_pro, xline_pro, ilines_all, 4), dtype='float32')

X_test_blocks[:, :, :, 0] = X_test[0:nt_pro, 0:xline_pro, :]
X_test_blocks[:, :, :, 1] = X_test[0:nt_pro, (xlines_all - xline_pro):, :]

X_test_blocks[:, :, :, 2] = X_test[(nt_all - nt_pro):, 0:xline_pro, :]
X_test_blocks[:, :, :, 3] = X_test[(nt_all - nt_pro):, (xlines_all - xline_pro):, :]
print('X_test_blocks.shape:' + str(X_test_blocks.shape))

eval_scores = np.zeros(shape=(1, 5, len(model_number_check)), dtype=np.float32)
eval_scores_classes = np.zeros(shape=(1, 2, n_class, len(model_number_check)), dtype=np.float32)

for i_model in model_number_check:
    model_number = i_model
    model_id = models_path + args.model_folder_id + '/model_' + format(model_number, '03d') + '.hdf5'
    print(model_id)
    model = load_model(model_id, compile=False)
    Y_pred_blocks = np.zeros(shape=X_test_blocks.shape, dtype='float32')
    for i_block in range(X_test_blocks.shape[-1]):
        for i_iline in range(ilines_all):
            X_test_temp = np.reshape(X_test_blocks[:, :, i_iline, i_block], (1, nt_pro, xline_pro, 1))
            start_time = time.time()
            Y_pred_temp = model.predict(X_test_temp)
            if args.Y_1 == 1:
                Y_pred_temp = Ypred_621(Y_pred_temp)
            else:
                Y_pred_temp = ReClassification(Y_pred_temp)
            elapsed_time = time.time() - start_time
            Y_pred_blocks[:, :, i_iline, i_block] = np.reshape(Y_pred_temp, (nt_pro, xline_pro))
            print('Model' + format(model_number, '03d') + '_block' + format(i_block + 1, '01d') + '_section ' + format(i_iline + 1, '04d') + ' done! All ' + format(ilines_all, '04d') + '. Remain ' + format(
                ilines_all - (i_iline + 1), '04d') + '.' + ' time=%4.3fs' % (elapsed_time))
            del X_test_temp, Y_pred_temp, start_time, elapsed_time
        print('')

    del model_number, model_id, model
    Y_pred = data_merge421(Y_pred_blocks, nt_all, xlines_all, nt_pro, xline_pro)
    del Y_pred_blocks
    print(Y_pred.shape)
    if args.iline_way != 1:
        Y_pred = Y_pred.transpose((0, 2, 1))
    print(Y_pred.shape)
    Y_pred.transpose((2, 1, 0)).astype('float32').tofile(
        paper_results_big_dir + '/' + data_name_prefix + '_' + 'Y_pred' + '_' + str(Y_pred.shape[0]) + 'x' + str(Y_pred.shape[1]) + 'x' + str(Y_pred.shape[2]) + '_model' + str(i_model) + '.bin')

    start_time = time.time()
    eval_scores[0, 0, i_model - model_number_check[0]] = snr(Y_true_available[:, 1:, :], Y_pred[:, 1:xlines_true, 0:ilines_true])
    accuracy_over_all_classes, accuracy_for_a_class, average_class_accuracy, IoU, average_IoU, weighted_IoU = get_classification_scores(Y_true_available[:, 1:, :], Y_pred[:, 1:xlines_true, 0:ilines_true], n_class)
    eval_scores[0, 1, i_model - model_number_check[0]] = accuracy_over_all_classes
    eval_scores[0, 2, i_model - model_number_check[0]] = average_class_accuracy
    eval_scores[0, 3, i_model - model_number_check[0]] = average_IoU
    eval_scores[0, 4, i_model - model_number_check[0]] = weighted_IoU

    eval_scores_classes[0, 0, :, i_model - model_number_check[0]] = accuracy_for_a_class
    eval_scores_classes[0, 1, :, i_model - model_number_check[0]] = IoU

    elapsed_time = time.time() - start_time
    print(str(elapsed_time))

    print(str(eval_scores))
    print(str(eval_scores_classes))

    eval_scores.transpose((2, 1, 0)).astype('float32').tofile(
        paper_results_save_dir + '/' + data_name_prefix+ '_models' + str(args.model_number_begin) + '_' + str(args.model_number_end) + '_cube_eval_scores' + '_' + str(eval_scores.shape[0]) + 'x' + str(
            eval_scores.shape[1]) + 'x' + str(eval_scores.shape[2]) + '.bin')

    eval_scores_classes.transpose((3, 2, 1, 0)).astype('float32').tofile(
        paper_results_save_dir + '/' + data_name_prefix + '_models' + str(args.model_number_begin) + '_' + str(args.model_number_end) + '_cube_eval_scores_classes' + '_' + str(
            eval_scores_classes.shape[0]) + 'x' + str(eval_scores_classes.shape[1]) + 'x' + str(eval_scores_classes.shape[2]) + 'x' + str(eval_scores_classes.shape[3]) + '.bin')

    del Y_pred
    ################################################################################
    fig = plt.figure()
    fig.set_size_inches(19.2, 10.80)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig.add_subplot(1, 2, 1)
    plt.plot(eval_scores[:, 0, :].transpose((1, 0)))
    fig.add_subplot(1, 2, 2)
    plt.plot(eval_scores[:, 3, :].transpose((1, 0)))
    fname = figures_dir + '/' + args.model_folder_id + '_cube_eval_scores' + '.jpg'
    if figures_save:
        plt.savefig(fname=fname, dpi=100, facecolor='w', edgecolor='w')
    print(' ')

if args.plot_show:
    plt.show()
