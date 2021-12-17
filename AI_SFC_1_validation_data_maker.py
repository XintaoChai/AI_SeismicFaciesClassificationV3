from pyseismic_local.date_time_functions import *  # 自己写的日期时间函数
from pyseismic_local.AI_SeismicFaciesClassification import *  # 自己写的画图相关函数

import time  # 加载时间库

import numpy as np  # 加载numpy
import segyio

################################################################################
data_time_now_str = data_time_str_def()  # 获取当前的时间戳，保存数据用

sample_channels = 1  # 灰度图片或地震数据通道为1，RGB图片数据通道为3
np.random.seed(seed=0)  # for reproducibility随机数可重复，类似MATLAB的rng('default')
random.seed(0)
plot_show = True  # 是否画图查看
patch_rows = 992
patch_cols = 576
left_for_test_iline = 7
left_for_test_xline = 45
stride = [230, 14]
data_save = 1
Y_channels = 1
################################################################################
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 做数据只使用CPU即可，用不到GPU
print('CPU is used.')

################################################################################
disk_ID = 'G'  # 数据存到哪个硬盘？这里存到了D盘某个文件夹，像这么多样本，不建议存在当前文件夹
module_name = 'AI_SeismicFaciesClassification'  # 所研发模块的名字

module_data_dir = disk_ID + ':/BigData/' + module_name  # 模块数据文件夹，存放了训练数据和网络模型
if not os.path.exists(module_data_dir):
    os.mkdir(module_data_dir)

validation_data_dir = module_data_dir + '/' + 'validation_data_' + str(int(patch_rows)) + 'x' + str(
    int(patch_cols)) + '_Ychannels' + str(Y_channels)
if not os.path.exists(validation_data_dir):
    os.mkdir(validation_data_dir)
################################################################################
raw_data_X_ID = []
raw_data_Y_ID = []
patch_stride = []

raw_data_X_ID.append('D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' + 'TrainingData_Image.segy')
raw_data_Y_ID.append('D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' + 'TrainingData_Labels.segy')
patch_stride.append(stride[0])

raw_data_X_ID.append('D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' + 'TrainingData_Image.segy')
raw_data_Y_ID.append('D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' + 'TrainingData_Labels.segy')
patch_stride.append(stride[1])

raw_data_X_ID.append('D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' + 'TrainingData_Image.segy')
raw_data_Y_ID.append('D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' + 'TrainingData_Labels.segy')
patch_stride.append(stride[0])

raw_data_X_ID.append('D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' + 'TrainingData_Image.segy')
raw_data_Y_ID.append('D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' + 'TrainingData_Labels.segy')
patch_stride.append(stride[1])

################################################################################
if __name__ == '__main__':
    dataset_number = len(raw_data_X_ID)
    counter = 0
    start_time = time.time()
    for i in range(dataset_number):
        counter_per_data = 0
        print('dataID ' + str(i + 1) + '/' + str(dataset_number) + ': ' + raw_data_X_ID[i])
        ########################################################################
        # Read data cube
        X = segyio.tools.cube(raw_data_X_ID[i])
        Y = segyio.tools.cube(raw_data_Y_ID[i])
        print('X.shape:' + str(X.shape))
        print('Y.shape:' + str(Y.shape))
        X = X.transpose((2, 1, 0)).astype('float32')
        Y = Y.transpose((2, 1, 0)).astype('float32')
        print('X.shape:' + str(X.shape))
        print('Y.shape:' + str(Y.shape))
        if i == 1:
            X = X[1006 - 992:, -left_for_test_xline:, :]
            Y = Y[1006 - 992:, -left_for_test_xline:, :]
        elif i == 0:
            X = X[1006 - 992:, 1:, -left_for_test_iline:]
            Y = Y[1006 - 992:, 1:, -left_for_test_iline:]
        elif i == 3:
            X = X[1006 - 992:, 1:left_for_test_xline, :]
            Y = Y[1006 - 992:, 1:left_for_test_xline, :]
        elif i == 2:
            X = X[1006 - 992:, 1:, 0:left_for_test_iline]
            Y = Y[1006 - 992:, 1:, 0:left_for_test_iline]
        print('X.shape:' + str(X.shape))
        print('Y.shape:' + str(Y.shape))
        ########################################################################
        if Y_channels > 1:
            Y_N = np.zeros(shape=[Y.shape[0], Y.shape[1], Y.shape[2], Y_channels], dtype='float32')
            for facies_class in range(Y_channels):
                p = np.where(Y == (facies_class + 1))
                Y_N[p[0], p[1], p[2], facies_class] = 1
                del p
            del Y
            Y = Y_N
            del Y_N
        ########################################################################
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
        Y = np.reshape(Y, (Y.shape[0], Y.shape[1], Y.shape[2], Y_channels))
        print('X.shape:' + str(X.shape))
        print('Y.shape:' + str(Y.shape))
        if i == 0 or i == 2:
            X = X.transpose((2, 0, 1, 3))
            Y = Y.transpose((2, 0, 1, 3))
        else:
            X = X.transpose((1, 0, 2, 3))
            Y = Y.transpose((1, 0, 2, 3))
        print('X.shape:' + str(X.shape))
        print('Y.shape:' + str(Y.shape))

        X = data2patches(X, patch_rows, patch_cols, patch_stride[i], patch_stride[i])
        print('X.shape:' + str(X.shape))
        Y = data2patches(Y, patch_rows, patch_cols, patch_stride[i], patch_stride[i])
        print('Y.shape:' + str(Y.shape))
        ########################################################################
        if plot_show:
            plot_1st_rand_lst_XandY_SFC(X, Y)
        ########################################################################
        sample_number_temp = X.shape[0]
        if data_save:
            for j in range(sample_number_temp):
                current_data_name_ID = counter + j + 1
                X_temp = np.squeeze(X[j])
                Y_temp = np.squeeze(Y[j])
                X_temp = X_temp.transpose((1, 0))
                if Y_channels > 1:
                    Y_temp = Y_temp.transpose((1, 0, 2))
                else:
                    Y_temp = Y_temp.transpose((1, 0))

                X_temp.astype('float32').tofile(
                    validation_data_dir + '/' + format(current_data_name_ID, '011d') + 'X.bin')
                Y_temp.astype('float32').tofile(
                    validation_data_dir + '/' + format(current_data_name_ID, '011d') + 'Y.bin')

                if current_data_name_ID % 500 == 0 or j == sample_number_temp - 1:
                    print('Writing ' + format(current_data_name_ID, '011d') + ' done!')
                del X_temp, Y_temp
        counter_per_data = counter_per_data + sample_number_temp
        counter = counter + sample_number_temp
        print('counter_per_data ' + str(counter_per_data))
        print('counter ' + str(counter))
        del sample_number_temp
        ########################################################################
        elapsed_time = time.time() - start_time
        elapsed_time_str = time2HMS(elapsed_time=elapsed_time)

        print('                 ')

    if plot_show:
        plt.show()
