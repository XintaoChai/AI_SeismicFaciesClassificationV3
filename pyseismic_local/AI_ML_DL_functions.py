import os
import glob  # glob 文件名模式匹配，不用遍历整个目录判断每个文件是不是符合。
# glob模块用来查找文件目录和文件，glob支持*?[]这三种通配符
import re  # regular-expression 正则表达式用来处理字符串
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def findLastCheckpoint(model_log_dir):
    # glob.glob返回所有匹配的文件路径列表。
    # 它只有一个参数pathname，定义了文件路径匹配规则，
    # 这里可以是绝对路径，也可以是相对路径。
    file_list = glob.glob(os.path.join(model_log_dir, 'model_*.hdf5'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            # 正则 re.findall  的用法
            # （返回string中所有与pattern相匹配的全部字串，返回形式为数组）
            # 语法：findall(pattern, string, flags=0)
            result = re.findall(".*model_(.*).hdf5.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def plot_training_validation_logs(log_csv_id):
    ################################################################################
    ################################################################################
    # pd.read_csv(filename)：从CSV文件导入数据
    log_csv = pd.read_csv(log_csv_id, header=0)
    fig = plt.figure()
    fig.set_size_inches(19.20, 10.80)
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig.add_subplot(2, 2, 1)
    plt.plot(log_csv['epoch'].values[0:] + 1, np.log10(log_csv['loss'].values[0:] / log_csv['loss'].values[0]))
    plt.ylim((-3.11, 0.1))
    plt.ylabel('loss (normalized)')
    plt.xticks(log_csv['epoch'].values[::5] + 1)
    plt.title('(a)训练过程中的损失函数变化趋势')

    fig.add_subplot(2, 2, 2)
    plt.plot(log_csv['epoch'].values[0:] + 1, log_csv['accuracy'].values[0:])
    plt.ylim((-0.1, 1.1))
    plt.ylabel('accuracy预测精度')
    plt.xticks(log_csv['epoch'].values[::5] + 1)
    plt.title('(b)训练过程中的预测精度变化趋势')

    fig.add_subplot(2, 2, 3)
    plt.plot(log_csv['epoch'].values[0:] + 1, np.log10(log_csv['val_loss'].values[0:] / log_csv['val_loss'].values[0]))
    plt.ylim((-3.11, 0.1))
    plt.xlabel('epoch 轮数')
    plt.xticks(log_csv['epoch'].values[::5] + 1)
    plt.ylabel('val loss (normalized)')
    plt.title('(c)Validation校验过程中的损失函数变化趋势')

    fig.add_subplot(2, 2, 4)
    plt.plot(log_csv['epoch'].values[0:] + 1, log_csv['val_accuracy'].values[0:])
    plt.ylim((-0.1, 1.1))
    plt.xlabel('epoch 轮数')
    plt.xticks(log_csv['epoch'].values[::5] + 1)
    plt.ylabel('val accuracy校验精度')
    plt.title('(d)Validation校验过程中的预测精度变化趋势')
