import numpy as np


def bin_data_reader2D_float32(data_id, sample_height, sample_width,
                              sample_channels, sample_number):
    data = np.fromfile(data_id, dtype='float32')
    # data = np.fromfile(data_id, dtype='float32')
    # print(str(data.shape))
    data = np.reshape(data, (
        sample_number, sample_width, sample_height, sample_channels))
    data = data.transpose((0, 2, 1, 3))
    # print('data.shape: ' + str(data.shape))
    return data

def bin_data_reader3D_float32(data_id, sample_height, sample_width,
                              sample_third, sample_channels, sample_number):
    data = np.fromfile(data_id, dtype='float32')
    # np.fromfile(frame, dtype=float)
    # frame: 文件、字符串
    # dtype: 读取的数据类型 。
    # 我们读取数据的时候都需要指定数据类型，无论是不是一维二维。默认为浮点型
    data = np.reshape(data,
                      (sample_number, sample_third, sample_width, sample_height,
                       sample_channels))
    data = data.transpose((0, 3, 2, 1, 4))
    # print('data.shape: ' + str(data.shape))
    return data

def txt_data_writer(file_name, data):
    txt_file = open(file_name, 'w')
    print(data, file=txt_file)
    txt_file.close()
