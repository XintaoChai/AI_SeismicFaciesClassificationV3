import numpy as np


def data2patches(data, patch_size_height, patch_size_width, patch_stride_height,
                 patch_stride_width):
    sample_number = data.shape[0]
    sample_height = data.shape[1]
    sample_width = data.shape[2]
    sample_channels = data.shape[3]
    data_patches = []
    for i in range(0, sample_number, 1):
        data_patches_temp = gen_patches(data[i, :, :, :], sample_height,
                                        sample_width,
                                        patch_size_height, patch_size_width,
                                        patch_stride_height,
                                        patch_stride_width)
        data_patches.append(data_patches_temp)

    data_patches = np.array(data_patches, dtype='float32')
    print('data_patches.shape: ' + str(data_patches.shape))
    data_patches = data_patches.reshape(
        (sample_number * data_patches.shape[1], patch_size_height,
         patch_size_width, sample_channels))
    print('len(data_patches): ' + str(len(data_patches)))

    return data_patches


def gen_patches(data_per_sample, sample_height, sample_width, patch_size_height,
                patch_size_width,
                patch_stride_height,
                patch_stride_width):
    data_patch = []
    # extract data_patch
    # print(str(data_per_sample.shape))
    patch_indexes_height, patch_indexes_width, patch_number_height, patch_number_width = patch_indexes(
        sample_height, sample_width,
        patch_size_height,
        patch_size_width,
        patch_stride_height,
        patch_stride_width)
    # print('patch_number_height: ' + str(patch_number_height))
    # print('patch_number_width: ' + str(patch_number_width))

    for i in range(0, patch_number_height, 1):
        for j in range(0, patch_number_width, 1):
            x = data_per_sample[patch_indexes_height[i]:patch_indexes_height[
                                                            i] + patch_size_height,
                patch_indexes_width[j]:patch_indexes_width[
                                           j] + patch_size_width, :]
            data_patch.append(x)
    # print(str(data_patch.shape))
    return data_patch


def patch_indexes(sample_height, sample_width, patch_size_height,
                  patch_size_width,
                  patch_stride_height, patch_stride_width):
    patch_indexes_height = list(
        range(0, sample_height - patch_size_height + 1, patch_stride_height))
    patch_indexes_width = list(
        range(0, sample_width - patch_size_width + 1, patch_stride_width))

    if patch_indexes_height[-1] + patch_size_height < sample_height:
        patch_indexes_height.append(sample_height - patch_size_height)

    if patch_indexes_width[-1] + patch_size_width < sample_width:
        patch_indexes_width.append(sample_width - patch_size_width)

    patch_number_height = len(patch_indexes_height)
    patch_number_width = len(patch_indexes_width)
    return patch_indexes_height, patch_indexes_width, patch_number_height, patch_number_width