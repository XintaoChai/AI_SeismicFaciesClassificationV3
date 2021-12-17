from pyseismic_local.AI_ML_DL_functions import *
from pyseismic_local.date_time_functions import *
from pyseismic_local.AI_SeismicFaciesClassification import *
from pyseismic_local.models_networks_ANNs.FlexibleBridgeNet_Keras import *
from pyseismic_local.public_functions import *

import argparse
import time

import numpy as np

from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.optimizers import Adam

################################################################################
data_time_str = data_time_str_def()

X_channels = 1
np.random.seed(seed=0)  # for reproducibility
random.seed(0)
# Params
parser = argparse.ArgumentParser()
# Common
parser.add_argument('--gpuID', default=0, type=int, help='gpuID')
parser.add_argument('--training_data_disk', nargs='?', type=str, default='D', help='training_data_disk')
parser.add_argument('--models_disk', nargs='?', type=str, default='F', help='models_disk')
parser.add_argument('--training_number', default=10000, type=int, help='training_number')
parser.add_argument('--epochs', default=3, type=int, help='epochs')
parser.add_argument('--batch_size', default=16, type=int, help='batch_size')

parser.add_argument('--loss_used', default=5, type=int, help='loss_used')
parser.add_argument('--lr', default=0.000100, type=float, help='lr')
parser.add_argument('--kernel_size', default=11, type=int, help='kernel_size')
parser.add_argument('--BridgeNet_used', default=5, type=int, help='BridgeNet_used')

# Mostly unchanged
parser.add_argument('--nD', default=2, type=int, help='nD')
parser.add_argument('--continue_train', action='store_true', help='continue_train')
parser.add_argument('--save_every', default=1, type=int, help='save_every')
parser.add_argument('--patch_rows', default=992, type=int, help='patch_rows')
parser.add_argument('--patch_cols', default=576, type=int, help='patch_cols')
parser.add_argument('--stride', default=[15, 5], type=int, help='stride', nargs='*')
parser.add_argument('--Y_channels', default=1, type=int, help='Y_channels')
parser.add_argument('--Y_channels_model', default=1, type=int, help='Y_channels_model')
parser.add_argument('--X_normal', default=5522.086, type=float, help='X_normal')
parser.add_argument('--plot_show', action='store_true', help='plot_show')
parser.add_argument('--kernels_all', default=[16, 32, 64, 128, 256, 512], type=int, nargs='*')
parser.add_argument('--conv2act_repeat', default=2, type=int, help='conv2act_repeat')
parser.add_argument('--reproduce', default=1, type=int, help='reproduce')
parser.add_argument('--res_case', default=0, type=int, help='res_case')
parser.add_argument('--res_number', default=0, type=int, help='res_number')
args = parser.parse_args()
print('args.loss_used: ' + str(args.loss_used))
if args.loss_used == 5:
    Y_1 = 1
    args.act_last = 3
    args.Y_channels_model = 6
else:
    Y_1 = 0
print('args.act_last: ' + str(args.act_last))
print('args.Y_channels_model: ' + str(args.Y_channels_model))
################################################################################
import os

if args.gpuID == -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('CPU is used.')
elif args.gpuID == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('GPU device ' + str(args.gpuID) + ' is used.')
elif args.gpuID == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('GPU device ' + str(args.gpuID) + ' is used.')

################################################################################
print(args.training_data_disk)
print(args.models_disk)
disk_ID_data = args.training_data_disk
disk_ID_model = args.models_disk
module_name = 'AI_SeismicFaciesClassification'
module_data_dir = disk_ID_data + ':/BigData/' + module_name
module_model_dir = disk_ID_model + ':/BigData/' + module_name
if not os.path.exists(module_model_dir):
    os.mkdir(module_model_dir)
training_data_dir = module_data_dir + '/' + 'training_data_' + str(int(args.patch_rows)) + 'x' + str(int(args.patch_cols)) + '_stride' + str(args.stride[0]) + '_' + str(
    args.stride[1]) + '_Ychannels' + str(args.Y_channels)
training_data_path = training_data_dir + '/'
validation_data_dir = module_data_dir + '/' + 'validation_data_' + str(int(args.patch_rows)) + 'x' + str(int(args.patch_cols)) + '_Ychannels' + str(args.Y_channels)
validation_data_path = validation_data_dir + '/'

models_dir = module_model_dir + '/' + 'models'
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

################################################################################
model_folder_name = f'patchR{args.patch_rows:04d}C{args.patch_cols:04d}_batch{args.batch_size:03d}_number{args.training_number}_stride{args.stride[0]}_{args.stride[1]}' \
                    f'_lr{round(args.lr * 10000000)}_kSize{args.kernel_size:02d}_ResCase{args.res_case}_ResNo{args.res_number:01d}_Net{args.BridgeNet_used:01d}_Crep{args.conv2act_repeat:01d}' \
                    f'_Repro{args.reproduce:02d}' + '_kernels' + (str(args.kernels_all[0:(args.BridgeNet_used + 1)])[1:-1]).replace(', ', '_')

print(model_folder_name)

model_log_dir = models_dir + '/' + model_folder_name

if not os.path.exists(model_log_dir):
    os.mkdir(model_log_dir)


def lr_scheduler(epoch):
    initial_lr = args.lr
    lr = initial_lr

    log('current learning rate is %2.11f' % lr)
    return lr


if __name__ == '__main__':
    ############################################################################
    file_suffix = 'bin'
    available_training_number = findFileNumber(training_data_path, file_suffix)
    print('available_training_number: ' + str(available_training_number))
    available_training_number = int(available_training_number / 2)
    print('available_training_number: ' + str(available_training_number))
    if available_training_number == 0:
        print('Error! No training data there!')
        exit(1)
    elif available_training_number > 0 and available_training_number < args.training_number:
        print('Warning! available sample number < training number!')
        args.training_number = available_training_number

    actual_training_number = (args.training_number // args.batch_size) * args.batch_size
    print('actual_training_number: ' + str(actual_training_number))

    steps_per_epoch = (actual_training_number // args.batch_size)
    txt_data_writer(model_log_dir + '/' + data_time_str + '_arguments' + '.txt', args)

    ############################################################################
    available_validation_number = findFileNumber(validation_data_path, file_suffix)
    available_validation_number = int(available_validation_number / 2)
    print('available_validation_number: ' + str(available_validation_number))
    if available_validation_number == 0:
        print('Error! No validation data there!')
    validation_X = np.zeros(shape=[available_validation_number, args.patch_rows, args.patch_cols, X_channels], dtype='float32')
    validation_Y = np.zeros(shape=[available_validation_number, args.patch_rows, args.patch_cols, args.Y_channels], dtype='float32')

    for j in range(available_validation_number):
        validation_X_dataID = validation_data_dir + '/' + format(j + 1, '011d') + 'X.bin'
        validation_Y_dataID = validation_data_dir + '/' + format(j + 1, '011d') + 'Y.bin'
        validation_X_data = bin_data_reader2D_float32(validation_X_dataID, args.patch_rows, args.patch_cols, X_channels, 1)
        validation_Y_data = bin_data_reader2D_float32(validation_Y_dataID, args.patch_rows, args.patch_cols, args.Y_channels, 1)
        validation_X[j] = validation_X_data[0]
        validation_Y[j] = validation_Y_data[0]
        '''
        print('validation_X min: ' + str(            np.min(validation_X[j])) + ' max: ' + str(np.max(validation_X[j])))
        print('validation_Y min: ' + str(            np.min(validation_Y[j])) + ' max: ' + str(np.max(validation_Y[j])))
        '''
        del validation_X_dataID, validation_X_data, validation_Y_dataID, validation_Y_data

    validation_X = validation_X / args.X_normal
    if Y_1:
        validation_Y = validation_Y - 1
    if args.plot_show:
        plot_1st_rand_lst_XandY_SFC(validation_X, validation_Y)

    if args.plot_show:
        plt.show()
    ############################################################################
    start_time = time.time()
    model = FlexibleBridgeNet(up_down_times=args.BridgeNet_used, Y_channels=args.Y_channels_model, kernel_size=args.kernel_size,
                              kernels_all=args.kernels_all[0:(args.BridgeNet_used + 1)], conv2act_repeat=args.conv2act_repeat, res_case=args.res_case,
                              res_number=args.res_number)

    print('kernels_all: ' + str(args.kernels_all[0:(args.BridgeNet_used + 1)]))

    model.summary()

    ############################################################################
    if args.continue_train:
        initial_epoch = findLastCheckpoint(model_log_dir=model_log_dir)
        if initial_epoch > 0:
            print('resuming by loading epoch %03d' % initial_epoch)
            model = load_model(os.path.join(model_log_dir, 'model_%03d.hdf5' % initial_epoch), compile=False)
    else:
        initial_epoch = 0

    ############################################################################
    # Regression Loss Function
    if args.loss_used == 1:
        loss_used = 'mean_squared_error'
    elif args.loss_used == 2:
        loss_used = 'mean_absolute_error'
    elif args.loss_used == 3:
        loss_used = 'mean_absolute_percentage_error'
    elif args.loss_used == 4:
        loss_used = 'mean_squared_logarithmic_error'
    # Multi-class Classification Loss
    elif args.loss_used == 5:
        loss_used = 'sparse_categorical_crossentropy'
        # 如果y_pred是经过softmax函数之后的输出，则from_logits=False
    elif args.loss_used == 6:
        loss_used = 'categorical_crossentropy'

    # 在Keras中，compile主要完成损失函数和优化器的一些配置，是为训练服务的。
    model.compile(optimizer=Adam(lr=args.lr), loss=loss_used, metrics=['accuracy'])
    # optimizer设定所选用的优化器函数
    # loss设定所选用的损失函数
    # metrics设定训练过程中的评价准则

    # use call back functions
    checkpointer = ModelCheckpoint(os.path.join(model_log_dir, 'model_{epoch:03d}.hdf5'), verbose=1, period=args.save_every)
    csv_logger = CSVLogger(os.path.join(model_log_dir, 'log.csv'), append=True, separator=',')
    lr_scheduler = LearningRateScheduler(lr_scheduler)
    tensorboard = TensorBoard(log_dir=model_log_dir, histogram_freq=0, batch_size=2, write_graph=True, write_grads=True, write_images=True)

    # 该过程来启动人工神经网络的训练
    history = model.fit_generator(
        training_data_generator_2D(args.epochs, args.batch_size, training_data_path, available_training_number,
                                   actual_training_number, args.patch_rows, args.patch_cols, X_channels,
                                   args.Y_channels, Y_1, args.X_normal, model_log_dir), steps_per_epoch=steps_per_epoch,
        epochs=args.epochs, verbose=1, initial_epoch=initial_epoch,
        callbacks=[checkpointer, csv_logger, lr_scheduler, tensorboard], validation_data=(validation_X, validation_Y))
    # epochs 设定人工神经网络训练多少轮，即看多少遍训练数据
    # batch_size 设定每次看多少个训练数据
    # training_data_path 训练数据存放的路径
    # actual_training_number 设定训练样本数
    # patch_rows 设定样本数据的高
    # patch_cols 设定样本数据的宽
    # steps_per_epoch 在每一轮训练过程中要迭代多少步
    # verbose 是否在控制台显示训练信息
    # callbacks 中设定了训练的人工神经网络存储、训练日志、学习率

    elapsed_time = time.time() - start_time
    elapsed_time_str = time2HMS(elapsed_time=elapsed_time)

    with open(os.path.join(model_log_dir, data_time_str + '_elapsed_time_' + elapsed_time_str + '.txt'), "w") as f:
        f.write(str(elapsed_time) + ' seconds    = ' + elapsed_time_str)
