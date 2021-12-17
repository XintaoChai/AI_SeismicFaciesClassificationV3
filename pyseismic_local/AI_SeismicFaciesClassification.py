import matplotlib.pyplot as plt
from pyseismic_local.data_read_write_functions import *
from pyseismic_local.data_processing_functions import *
from pyseismic_local.evaluation_metrics import *
import matplotlib  # 加载matplotlib库

import random  # 加载matplotlib库


# training_data_generator训练数据生成器
def training_data_generator_2D(epochs, batch_size, training_data_path, available_training_number,
                               actual_training_number, patch_size_height, patch_size_width, X_channels, Y_channels, Y_1,
                               X_normal, model_log_dir):
    while True:
        training_indexes_available = list(range(available_training_number))
        training_indexes_record = np.zeros(shape=(epochs, available_training_number), dtype=np.float32)
        for epoch in range(epochs):
            # print(epoch)
            np.random.shuffle(training_indexes_available)
            training_indexes = training_indexes_available[0:actual_training_number]
            training_indexes_record[epoch, training_indexes] = 1
            # print(training_indexes)
            if epoch == (epochs - 1):
                training_indexes_record.transpose((1, 0)).astype('float32').tofile(
                    model_log_dir + '/' + 'training_indexes_record' + '_' + str(
                        training_indexes_record.shape[0]) + 'x' + str(training_indexes_record.shape[1]) + '_' + str(
                        actual_training_number) + '.bin')
            # print('training_indexes:' + str(np.sort(training_indexes)))
            for i in range(0, len(training_indexes), batch_size):
                indexes_now = training_indexes[i:i + batch_size]
                training_X_per_batch = np.zeros(shape=[batch_size, patch_size_height, patch_size_width, X_channels],
                                                dtype=np.float32)
                training_Y_per_batch = np.zeros(shape=[batch_size, patch_size_height, patch_size_width, Y_channels],
                                                dtype=np.float32)
                for j in range(batch_size):
                    training_X_dataID = training_data_path + format(indexes_now[j] + 1, '011d') + 'X.bin'
                    training_Y_dataID = training_data_path + format(indexes_now[j] + 1, '011d') + 'Y.bin'
                    training_X_data = bin_data_reader2D_float32(training_X_dataID, patch_size_height, patch_size_width,
                                                                X_channels, 1)
                    training_Y_data = bin_data_reader2D_float32(training_Y_dataID, patch_size_height, patch_size_width,
                                                                Y_channels, 1)
                    training_X_per_batch[j] = training_X_data[0]
                    training_Y_per_batch[j] = training_Y_data[0]
                    del training_X_dataID, training_X_data, training_Y_dataID, training_Y_data
                training_X_per_batch = training_X_per_batch / X_normal
                if Y_1:
                    training_Y_per_batch = training_Y_per_batch - 1
                yield training_X_per_batch, training_Y_per_batch


def test_data_generator2D(test_data, training_data_dir):
    if test_data == 0:
        testID = random.randint(1, 30166)
        testID = 1
        # print(testID)
        X_id = training_data_dir + '/' + format(testID, '011d') + 'X.bin'
        Y_id = training_data_dir + '/' + format(testID, '011d') + 'Y.bin'
        dataset_name = 'train'

    if test_data == 1:
        X_id = 'D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' + 'TrainingData_Image.segy'
        Y_id = 'D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' + 'TrainingData_Labels.segy'
        dataset_name = 'val_iline'

    if test_data == 2:
        X_id = 'D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' + 'TrainingData_Image.segy'
        Y_id = 'D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' + 'TrainingData_Labels.segy'
        dataset_name = 'val_xline'

    if test_data == 3:
        X_id = 'D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/Training_Test1_Test2_Image_1006x1116x841.bin'
        Y_id = ''
        dataset_name = 'All_iline'

    if test_data == 4:
        X_id = 'D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/Training_Test1_Test2_Image_1006x1116x841.bin'
        Y_id = ''
        dataset_name = 'All_xline'

    if test_data == 5:
        X_id = 'D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/TestData_Image1.segy'
        Y_id = ''
        dataset_name = 'Test1_iline'

    if test_data == 6:
        X_id = 'D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/TestData_Image1.segy'
        Y_id = ''
        dataset_name = 'Test1_xline'

    if test_data == 7:
        X_id = 'D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/TestData_Image2.segy'
        Y_id = ''
        dataset_name = 'Test2_iline'

    if test_data == 8:
        X_id = 'D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/TestData_Image2.segy'
        Y_id = ''
        dataset_name = 'Test2_xline'

    return X_id, Y_id, dataset_name


def data_merge421(X_test_blocks, nt_all, xlines_all, nt_pro, xline_pro):
    ilines_all = X_test_blocks.shape[2]

    X_test_AB_Overlap = 0.5 * (X_test_blocks[:, (xlines_all - xline_pro):xline_pro, :, 0] + X_test_blocks[:, 0:(xline_pro - (xlines_all - xline_pro)), :, 1])

    X_test_CD_Overlap = 0.5 * (X_test_blocks[:, (xlines_all - xline_pro):xline_pro, :, 2] + X_test_blocks[:, 0:(xline_pro - (xlines_all - xline_pro)), :, 3])

    X_test_AB = np.zeros(shape=(nt_pro, xlines_all, ilines_all), dtype='float32')
    X_test_CD = np.zeros(shape=(nt_pro, xlines_all, ilines_all), dtype='float32')

    X_test_AB[:, 0:(xlines_all - xline_pro), :] = X_test_blocks[:, 0:(xlines_all - xline_pro), :, 0]
    X_test_AB[:, (xlines_all - xline_pro):xline_pro, :] = X_test_AB_Overlap[:, :, :]
    X_test_AB[:, xline_pro:, :] = X_test_blocks[:, (xline_pro - (xlines_all - xline_pro)):, :, 1]

    X_test_CD[:, 0:(xlines_all - xline_pro), :] = X_test_blocks[:, 0:(xlines_all - xline_pro), :, 2]
    X_test_CD[:, (xlines_all - xline_pro):xline_pro, :] = X_test_CD_Overlap[:, :, :]
    X_test_CD[:, xline_pro:, :] = X_test_blocks[:, (xline_pro - (xlines_all - xline_pro)):, :, 3]

    X_test_ABCD = np.zeros(shape=(nt_all, xlines_all, ilines_all), dtype='float32')
    X_test_ABCD[0:(nt_all - nt_pro), :, :] = X_test_AB[0:(nt_all - nt_pro), :, :]
    X_test_ABCD[(nt_all - nt_pro):nt_pro, :, :] = 0.5 * (X_test_AB[(nt_all - nt_pro):nt_pro, :, :] + X_test_CD[0:(nt_pro - (nt_all - nt_pro)), :, :])
    X_test_ABCD[nt_pro:, :, :] = X_test_CD[(nt_pro - (nt_all - nt_pro)):, :, :]
    return X_test_ABCD


def plot_SFC_2D_results(X_test, Y_true, Y_pred, show_str, alphaX, alphaY):
    patch_size_height = X_test.shape[1]
    patch_size_width = X_test.shape[2]
    cmapX = 'gray'
    cmapY = matplotlib.colors.ListedColormap(['red', 'yellow', 'green', 'blue', 'fuchsia', 'brown'])
    zoom_X = 0.1
    norm_X = matplotlib.colors.Normalize(vmin=np.min(X_test[:]) * zoom_X, vmax=np.max(X_test[:]) * zoom_X)
    norm_Y = matplotlib.colors.Normalize(vmin=1, vmax=6)

    fig = plt.figure()
    fig.set_size_inches(19.20, 10.80)
    fig.add_subplot(1, 6, (1, 2))
    plt.imshow(X_test.reshape(patch_size_height, patch_size_width), cmap=cmapX, norm=norm_X, aspect='auto', alpha=alphaX)
    plt.imshow(Y_true.reshape(patch_size_height, patch_size_width), cmap=cmapY, norm=norm_Y, aspect='auto', alpha=alphaY)

    fig.add_subplot(1, 6, (3, 4))
    plt.imshow(X_test.reshape(patch_size_height, patch_size_width), cmap=cmapX, norm=norm_X, aspect='auto', alpha=alphaX)
    plt.imshow(Y_pred.reshape(patch_size_height, patch_size_width), cmap=cmapY, norm=norm_Y, aspect='auto', alpha=alphaY)
    plt.title(show_str)

    fig.add_subplot(1, 6, (5, 6))
    plt.imshow((Y_true - Y_pred).reshape(patch_size_height, patch_size_width), cmap=cmapY, aspect='auto')


def plot_1st_rand_lst_XandY_SFC(X, Y):
    patch_size_height = X.shape[1]
    patch_size_width = X.shape[2]
    nmid = np.random.randint(1, X.shape[0], [1, 1]) - 1
    cmapX = 'gray'
    cmapY = 'hsv'
    zoom_X = 0.1
    norm_X = matplotlib.colors.Normalize(vmin=np.min(X[:]) * zoom_X,
                                         vmax=np.max(X[:]) * zoom_X)
    norm_Y = matplotlib.colors.Normalize(vmin=np.min(Y[:]), vmax=np.max(Y[:]))

    fig = plt.figure()
    fig.set_size_inches(17.2, 10.0)
    fig.add_subplot(2, 3, 1)
    plt.imshow(X[0].reshape(patch_size_height, patch_size_width),
               cmap=cmapX, norm=norm_X, aspect='auto')
    fig.add_subplot(2, 3, 2)
    plt.imshow(X[nmid].reshape(patch_size_height, patch_size_width),
               cmap=cmapX, norm=norm_X, aspect='auto')
    fig.add_subplot(2, 3, 3)
    plt.imshow(X[-1].reshape(patch_size_height, patch_size_width),
               cmap=cmapX, norm=norm_X, aspect='auto')
    fig.add_subplot(2, 3, 4)
    plt.imshow(Y[0].reshape(patch_size_height, patch_size_width),
               cmap=cmapY, norm=norm_Y, aspect='auto')
    fig.add_subplot(2, 3, 5)
    plt.imshow(Y[nmid].reshape(patch_size_height, patch_size_width),
               cmap=cmapY, norm=norm_Y, aspect='auto')
    fig.add_subplot(2, 3, 6)
    plt.imshow(Y[-1].reshape(patch_size_height, patch_size_width),
               cmap=cmapY, norm=norm_Y, aspect='auto')


def Ypred_621(Y_pred):
    Y_pred_temp = np.zeros(shape=(Y_pred.shape[0], Y_pred.shape[1], Y_pred.shape[2], 1), dtype=np.float32)
    for i_1 in range(Y_pred.shape[1]):
        for i_2 in range(Y_pred.shape[2]):
            Y_pred_temp[0, i_1, i_2, 0] = np.argmax(Y_pred[0, i_1, i_2, :]) + 1
    del Y_pred
    Y_pred = Y_pred_temp
    del Y_pred_temp
    return Y_pred


def ReClassification(data_in):
    data_out = np.reshape(data_in, (np.product(data_in.shape), 1))
    # print(data_out.shape)
    for i in range(len(data_out)):
        data_out[i] = ReClassification_each(data_out[i])
    data_out = np.reshape(data_out, data_in.shape)
    return data_out


def ReClassification_each(class_in):
    if class_in < 1.5:
        class_out = 1
    elif class_in < 2.5:
        class_out = 2
    elif class_in < 3.5:
        class_out = 3
    elif class_in < 4.5:
        class_out = 4
    elif class_in < 5.5:
        class_out = 5
    else:
        class_out = 6
    return class_out
