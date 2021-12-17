from keras.models import *
from keras.layers import *


# act_hide *: tanh, softmax,selu(可伸缩的指数线性单元),Softplus,softsign,relu,sigmoid,hard_sigmoid,exponential,linear
# Advanced Activations: LeakyReLU(带泄漏的 ReLU), PReLU(参数化的 ReLU), elu(指数线性单元), ThresholdedReLU(带阈值的修正线性单元)
# act_hide 1: relu
# act_hide 2: LeakyReLU
# act_hide 3: PReLU
# act_hide 4: ELU
# act_hide 5: ThresholdedReLU

# act_last 1: sigmoid
# act_last 2: tanh
# act_last 3: softmax

# pool_way 1: MaxPooling
# pool_way 2: AveragePooling

def input_block(nD=2, X_channels=1, pool_size_all=2):
    if nD == 3:
        top_inputs = Input(shape=(None, None, None, X_channels))
        pool_size = (pool_size_all, pool_size_all, pool_size_all)
    elif nD == 2:
        top_inputs = Input(shape=(None, None, X_channels))
        pool_size = (pool_size_all, pool_size_all)
    elif nD == 1:
        top_inputs = Input(shape=(None, X_channels))
        pool_size = pool_size_all
    return top_inputs, pool_size


def conv2act_block(nD=2, kernel_size=3, use_BN=False, kernels_now=16, act_hide=1, drop_rate=0, conv2act_repeat=1, dilation_rate=1, input_layer=None):
    for repeat in range(conv2act_repeat + 1):
        if nD == 3:
            output_layer = Conv3D(filters=kernels_now, kernel_size=kernel_size, padding='same', dilation_rate=(dilation_rate, dilation_rate, dilation_rate))(input_layer)
        elif nD == 2:
            output_layer = Conv2D(filters=kernels_now, kernel_size=kernel_size, padding='same', dilation_rate=(dilation_rate, dilation_rate))(input_layer)
        elif nD == 1:
            output_layer = Conv1D(filters=kernels_now, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate)(input_layer)
        if drop_rate > 0:
            output_layer = Dropout(rate=drop_rate)(output_layer)
        if use_BN:
            output_layer = BatchNormalization()(output_layer)
        if act_hide == 1:
            output_layer = Activation('relu')(output_layer)
        elif act_hide == 2:
            output_layer = LeakyReLU(alpha=0.3)(output_layer)
        elif act_hide == 3:
            output_layer = PReLU(alpha_initializer='zeros')(output_layer)
        elif act_hide == 4:
            output_layer = ELU(alpha=1.0)(output_layer)
        elif act_hide == 5:
            output_layer = ThresholdedReLU(theta=1.0)(output_layer)
        input_layer = output_layer

    return output_layer


def pooling_block(nD=2, pool_size=None, pool_way=1, input_layer=None):
    if nD == 3:
        if pool_way == 1:
            output_layer = MaxPooling3D(pool_size=pool_size)(input_layer)
        elif pool_way == 2:
            output_layer = AveragePooling3D(pool_size=pool_size)(input_layer)
    elif nD == 2:
        if pool_way == 1:
            output_layer = MaxPooling2D(pool_size=pool_size)(input_layer)
        elif pool_way == 2:
            output_layer = AveragePooling2D(pool_size=pool_size)(input_layer)
    elif nD == 1:
        if pool_way == 1:
            output_layer = MaxPooling1D(pool_size=pool_size)(input_layer)
        elif pool_way == 2:
            output_layer = AveragePooling1D(pool_size=pool_size)(input_layer)
    return output_layer


def up_concatenate_block(nD=2, up_sampling_size=None, up_layer=None, concatenate_layer=None):
    # ##########################################################################
    if nD == 3:
        output_layer = concatenate([UpSampling3D(size=up_sampling_size)(up_layer), concatenate_layer], axis=-1)
    elif nD == 2:
        output_layer = concatenate([UpSampling2D(size=up_sampling_size)(up_layer), concatenate_layer], axis=-1)
    elif nD == 1:
        output_layer = concatenate([UpSampling1D(size=up_sampling_size)(up_layer), concatenate_layer], axis=-1)

    return output_layer


def output_block(nD=2, Y_channels=1, act_last=0, input_layer=None):
    # ##########################################################################
    if nD == 3:
        output_layer = Conv3D(filters=Y_channels, kernel_size=1)(input_layer)
    elif nD == 2:
        output_layer = Conv2D(filters=Y_channels, kernel_size=1)(input_layer)
    elif nD == 1:
        output_layer = Conv1D(filters=Y_channels, kernel_size=1)(input_layer)

    # ##########################################################################
    if act_last == 1:
        output_layer = Activation('sigmoid')(output_layer)
    elif act_last == 2:
        # [-1 1]
        output_layer = Activation('tanh')(output_layer)
    elif act_last == 3:
        # 多分类
        output_layer = Activation('softmax')(output_layer)

    return output_layer


################################################################################
def BridgeNet_5(nD=2, X_channels=1, Y_channels=1, kernel_size=3, pool_size_all=2, use_BN=False, kernels_all=None, act_hide=1, act_last=0, pool_way=1, drop_rate=0, conv2act_repeat=1, dilation_rate=1):
    # ##########################################################################
    top_inputs, pool_size = input_block(nD, X_channels, pool_size_all)
    up_sampling_size = pool_size

    ####################################################################################################################
    conv01 = conv2act_block(nD, kernel_size, use_BN, kernels_all[0], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=top_inputs)
    down01 = pooling_block(nD, pool_size, pool_way, input_layer=conv01)
    ####################################################################################################################
    conv02 = conv2act_block(nD, kernel_size, use_BN, kernels_all[1], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=down01)
    down02 = pooling_block(nD, pool_size, pool_way, input_layer=conv02)
    ####################################################################################################################
    conv03 = conv2act_block(nD, kernel_size, use_BN, kernels_all[2], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=down02)
    down03 = pooling_block(nD, pool_size, pool_way, input_layer=conv03)
    ####################################################################################################################
    conv04 = conv2act_block(nD, kernel_size, use_BN, kernels_all[3], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=down03)
    down04 = pooling_block(nD, pool_size, pool_way, input_layer=conv04)
    ####################################################################################################################
    conv05 = conv2act_block(nD, kernel_size, use_BN, kernels_all[4], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=down04)
    down05 = pooling_block(nD, pool_size, pool_way, input_layer=conv05)
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    conv06 = conv2act_block(nD, kernel_size, use_BN, kernels_all[5], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=down05)
    up1 = up_concatenate_block(nD, up_sampling_size, up_layer=conv06, concatenate_layer=conv05)
    ####################################################################################################################
    conv07 = conv2act_block(nD, kernel_size, use_BN, kernels_all[4], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=up1)
    up2 = up_concatenate_block(nD, up_sampling_size, up_layer=conv07, concatenate_layer=conv04)
    ####################################################################################################################
    conv08 = conv2act_block(nD, kernel_size, use_BN, kernels_all[3], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=up2)
    up3 = up_concatenate_block(nD, up_sampling_size, up_layer=conv08, concatenate_layer=conv03)
    ####################################################################################################################
    conv09 = conv2act_block(nD, kernel_size, use_BN, kernels_all[2], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=up3)
    up4 = up_concatenate_block(nD, up_sampling_size, up_layer=conv09, concatenate_layer=conv02)
    ####################################################################################################################
    conv10 = conv2act_block(nD, kernel_size, use_BN, kernels_all[1], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=up4)
    up5 = up_concatenate_block(nD, up_sampling_size, up_layer=conv10, concatenate_layer=conv01)
    ####################################################################################################################
    conv11 = conv2act_block(nD, kernel_size, use_BN, kernels_all[0], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=up5)

    conv12 = output_block(nD, Y_channels, act_last, input_layer=conv11)

    model = Model(inputs=[top_inputs], outputs=[conv12])

    return model


################################################################################
def BridgeNet_4(nD=2, X_channels=1, Y_channels=1, kernel_size=3, pool_size_all=2, use_BN=False, kernels_all=None, act_hide=1, act_last=0, pool_way=1, drop_rate=0, conv2act_repeat=1, dilation_rate=1):
    top_inputs, pool_size = input_block(nD, X_channels, pool_size_all)
    up_sampling_size = pool_size

    ####################################################################################################################
    conv01 = conv2act_block(nD, kernel_size, use_BN, kernels_all[0], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=top_inputs)
    down01 = pooling_block(nD, pool_size, pool_way, input_layer=conv01)

    ####################################################################################################################
    conv02 = conv2act_block(nD, kernel_size, use_BN, kernels_all[1], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=down01)
    down02 = pooling_block(nD, pool_size, pool_way, input_layer=conv02)
    ####################################################################################################################
    conv03 = conv2act_block(nD, kernel_size, use_BN, kernels_all[2], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=down02)
    down03 = pooling_block(nD, pool_size, pool_way, input_layer=conv03)
    ####################################################################################################################
    conv04 = conv2act_block(nD, kernel_size, use_BN, kernels_all[3], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=down03)
    down04 = pooling_block(nD, pool_size, pool_way, input_layer=conv04)
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    conv05 = conv2act_block(nD, kernel_size, use_BN, kernels_all[4], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=down04)
    up1 = up_concatenate_block(nD, up_sampling_size, up_layer=conv05, concatenate_layer=conv04)
    ####################################################################################################################
    conv06 = conv2act_block(nD, kernel_size, use_BN, kernels_all[3], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=up1)
    up2 = up_concatenate_block(nD, up_sampling_size, up_layer=conv06, concatenate_layer=conv03)
    ####################################################################################################################
    conv07 = conv2act_block(nD, kernel_size, use_BN, kernels_all[2], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=up2)
    up3 = up_concatenate_block(nD, up_sampling_size, up_layer=conv07, concatenate_layer=conv02)
    ####################################################################################################################
    conv08 = conv2act_block(nD, kernel_size, use_BN, kernels_all[1], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=up3)
    up4 = up_concatenate_block(nD, up_sampling_size, up_layer=conv08, concatenate_layer=conv01)
    ####################################################################################################################
    conv09 = conv2act_block(nD, kernel_size, use_BN, kernels_all[0], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=up4)

    conv10 = output_block(nD, Y_channels, act_last, input_layer=conv09)

    model = Model(inputs=[top_inputs], outputs=[conv10])

    return model


################################################################################
def BridgeNet_3(nD=2, X_channels=1, Y_channels=1, kernel_size=3, pool_size_all=2, use_BN=False, kernels_all=None, act_hide=1, act_last=0, pool_way=1, drop_rate=0, conv2act_repeat=1, dilation_rate=1):
    # ##########################################################################
    top_inputs, pool_size = input_block(nD, X_channels, pool_size_all)

    up_sampling_size = pool_size

    ####################################################################################################################
    conv01 = conv2act_block(nD, kernel_size, use_BN, kernels_all[0], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=top_inputs)
    down01 = pooling_block(nD, pool_size, pool_way, input_layer=conv01)
    ####################################################################################################################
    conv02 = conv2act_block(nD, kernel_size, use_BN, kernels_all[1], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=down01)
    down02 = pooling_block(nD, pool_size, pool_way, input_layer=conv02)
    ####################################################################################################################
    conv03 = conv2act_block(nD, kernel_size, use_BN, kernels_all[2], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=down02)
    down03 = pooling_block(nD, pool_size, pool_way, input_layer=conv03)
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    conv04 = conv2act_block(nD, kernel_size, use_BN, kernels_all[3], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=down03)
    up1 = up_concatenate_block(nD, up_sampling_size, up_layer=conv04, concatenate_layer=conv03)
    ####################################################################################################################
    conv05 = conv2act_block(nD, kernel_size, use_BN, kernels_all[2], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=up1)
    up2 = up_concatenate_block(nD, up_sampling_size, up_layer=conv05, concatenate_layer=conv02)
    ####################################################################################################################
    conv06 = conv2act_block(nD, kernel_size, use_BN, kernels_all[1], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=up2)
    up3 = up_concatenate_block(nD, up_sampling_size, up_layer=conv06, concatenate_layer=conv01)
    ####################################################################################################################
    conv07 = conv2act_block(nD, kernel_size, use_BN, kernels_all[0], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=up3)

    conv08 = output_block(nD, Y_channels, act_last, input_layer=conv07)

    model = Model(inputs=[top_inputs], outputs=[conv08])

    return model


def BridgeNet_2(nD=2, X_channels=1, Y_channels=1, kernel_size=3, pool_size_all=2, use_BN=False, kernels_all=None, act_hide=1, act_last=0, pool_way=1, drop_rate=0, conv2act_repeat=1, dilation_rate=1):
    # ##########################################################################
    top_inputs, pool_size = input_block(nD, X_channels, pool_size_all)

    up_sampling_size = pool_size

    ####################################################################################################################
    conv01 = conv2act_block(nD, kernel_size, use_BN, kernels_all[0], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=top_inputs)
    down01 = pooling_block(nD, pool_size, pool_way, input_layer=conv01)
    ####################################################################################################################
    conv02 = conv2act_block(nD, kernel_size, use_BN, kernels_all[1], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=down01)
    down02 = pooling_block(nD, pool_size, pool_way, input_layer=conv02)
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    conv03 = conv2act_block(nD, kernel_size, use_BN, kernels_all[2], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=down02)
    up1 = up_concatenate_block(nD, up_sampling_size, up_layer=conv03, concatenate_layer=conv02)
    ####################################################################################################################
    conv04 = conv2act_block(nD, kernel_size, use_BN, kernels_all[1], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=up1)
    up2 = up_concatenate_block(nD, up_sampling_size, up_layer=conv04, concatenate_layer=conv01)
    ####################################################################################################################
    conv05 = conv2act_block(nD, kernel_size, use_BN, kernels_all[0], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=up2)

    conv06 = output_block(nD, Y_channels, act_last, input_layer=conv05)

    model = Model(inputs=[top_inputs], outputs=[conv06])

    return model


def BridgeNet_1(nD=2, X_channels=1, Y_channels=1, kernel_size=3, pool_size_all=2, use_BN=False, kernels_all=None, act_hide=1, act_last=0, pool_way=1, drop_rate=0, conv2act_repeat=1, dilation_rate=1):
    # ##########################################################################
    top_inputs, pool_size = input_block(nD, X_channels, pool_size_all)

    up_sampling_size = pool_size

    ####################################################################################################################
    conv01 = conv2act_block(nD, kernel_size, use_BN, kernels_all[0], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=top_inputs)
    down01 = pooling_block(nD, pool_size, pool_way, input_layer=conv01)
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    conv02 = conv2act_block(nD, kernel_size, use_BN, kernels_all[1], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=down01)
    up1 = up_concatenate_block(nD, up_sampling_size, up_layer=conv02, concatenate_layer=conv01)
    ####################################################################################################################
    conv03 = conv2act_block(nD, kernel_size, use_BN, kernels_all[0], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=up1)

    conv04 = output_block(nD, Y_channels, act_last, input_layer=conv03)

    model = Model(inputs=[top_inputs], outputs=[conv04])

    return model
