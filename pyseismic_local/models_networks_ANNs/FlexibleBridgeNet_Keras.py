from keras.models import *
from keras.layers import *


# Taihui Yang
def FlexibleBridgeNet(up_down_times=4, nD=2, X_channels=1, Y_channels=1, kernel_size=3, pool_size_all=2, use_BN=0, kernels_all=None, act_hide=1, act_last='softmax',
                      pool_way='max', drop_rate=0, conv2act_repeat=2, dilation_rate=1, final_minus=0, res_case=0, res_number=1):
    # 不使用残差模块
    def conv_block0(filters):
        def f(input_layer):
            output_layer = input_layer
            for i in range(conv2act_repeat):
                output_layer = f_conv[0](filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=f_conv[1])(output_layer)
                if drop_rate > 0:
                    output_layer = Dropout(rate=drop_rate)(output_layer)
                if use_BN:
                    output_layer = BatchNormalization()(output_layer)
                if f_act_hide:
                    output_layer = f_act_hide[0](**f_act_hide[1])(output_layer)
            return output_layer

        return f

    # 激活+激活
    def conv_block2(filters):
        def f(input_layer):
            output_layer = input_layer
            for i in range(conv2act_repeat):
                output_layer = f_conv[0](filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=f_conv[1])(output_layer)
                if drop_rate > 0:
                    output_layer = Dropout(rate=drop_rate)(output_layer)
                if use_BN:
                    output_layer = BatchNormalization()(output_layer)
                if f_act_hide:
                    output_layer = f_act_hide[0](**f_act_hide[1])(output_layer)
                if i == 0:
                    input_layer = output_layer
            return add([input_layer, output_layer])

        return f

    # 卷积+卷积
    def conv_block3(filters):
        def f(input_layer):
            output_layer = input_layer
            for i in range(conv2act_repeat):
                output_layer = f_conv[0](filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=f_conv[1])(output_layer)
                if i == 0:
                    input_layer = output_layer
                if drop_rate > 0:
                    output_layer = Dropout(rate=drop_rate)(output_layer)
                if use_BN:
                    output_layer = BatchNormalization()(output_layer)
                if f_act_hide and i != conv2act_repeat - 1:
                    output_layer = f_act_hide[0](**f_act_hide[1])(output_layer)
            output_layer = add([input_layer, output_layer])
            if f_act_hide:
                output_layer = f_act_hide[0](**f_act_hide[1])(output_layer)
            return output_layer

        return f

    # 激活+卷积
    def conv_block4(filters):
        def f(input_layer):
            output_layer = input_layer
            for i in range(conv2act_repeat):
                output_layer = f_conv[0](filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=f_conv[1])(output_layer)
                if drop_rate > 0:
                    output_layer = Dropout(rate=drop_rate)(output_layer)
                if use_BN:
                    output_layer = BatchNormalization()(output_layer)
                if f_act_hide and i != conv2act_repeat - 1:
                    output_layer = f_act_hide[0](**f_act_hide[1])(output_layer)
                if i == 0:
                    input_layer = output_layer
            output_layer = add([input_layer, output_layer])
            if f_act_hide:
                output_layer = f_act_hide[0](**f_act_hide[1])(output_layer)
            return output_layer

        return f

    def res_block(filters, res_number=1):
        def f(input_layer):
            output_layer = input_layer
            for i in range(res_number):
                output_layer = conv_block_use(filters)(output_layer)
            return output_layer

        return f

    def down_block(filters):
        def f(input_layer):
            conved_layer = conv_block(filters)(input_layer)
            output_layer = f_pool[0](pool_size=f_pool[1])(conved_layer)
            return conved_layer, output_layer

        return f

    def up_block(filters, cat_layer):
        def f(input_layer):
            conved_layer = conv_block(filters)(input_layer)
            output_layer = concatenate([f_up[0](size=f_up[1])(conved_layer), cat_layer])
            return output_layer

        return f

    def out_block(filters: list):
        def f(input_layer):
            conved_layer = conv_block(filters[0])(input_layer)
            output_layer = f_conv[0](filters=filters[1], kernel_size=1)(conved_layer)
            if act_last in ['sigmoid', 'tanh', 'softmax']:
                output_layer = Activation(act_last)(output_layer)
            return output_layer

        return f

    conv_block_dict = {0: conv_block0,
                       2: conv_block2,
                       3: conv_block3,
                       4: conv_block4}
    conv_dict = {1: [Conv1D, dilation_rate],
                 2: [Conv2D, (dilation_rate, dilation_rate)],
                 3: [Conv3D, (dilation_rate, dilation_rate, dilation_rate)]}
    pool_dict = {'max': {1: [MaxPooling1D, pool_size_all],
                         2: [MaxPooling2D, (pool_size_all, pool_size_all)],
                         3: [MaxPooling3D, (pool_size_all, pool_size_all, pool_size_all)]},
                 'mean': {1: [AveragePooling1D, pool_size_all],
                          2: [AveragePooling2D, (pool_size_all, pool_size_all)],
                          3: [AveragePooling3D, (pool_size_all, pool_size_all, pool_size_all)]}}
    up_dict = {1: [UpSampling1D, pool_size_all],
               2: [UpSampling2D, (pool_size_all, pool_size_all)],
               3: [UpSampling3D, (pool_size_all, pool_size_all, pool_size_all)]}
    act_hide_dict = {1: [Activation, dict(activation='relu')],
                     2: [LeakyReLU, dict(alpha=0.3)],
                     3: [PReLU, dict(alpha_initializer='zeros')],
                     4: [ELU, dict(alpha=1.0)],
                     5: [ThresholdedReLU, dict(theta=1.0)]}
    ##########################################################################################
    if res_case:
        res_block.__defaults__ = res_number,
        conv_block_use = conv_block_dict[res_case]
        conv_block = res_block
    else:
        conv_block = conv_block_dict[res_case]
    if len(kernels_all) < up_down_times + 1:
        raise ValueError('len(kernels_all) < up_down_times + 1')
    elif len(kernels_all) > up_down_times + 1:
        kernels_all = kernels_all[:up_down_times + 1]
    f_conv = conv_dict[nD]
    f_up = up_dict[nD]
    if pool_way in ['max', 'mean']:
        f_pool = pool_dict[pool_way][nD]
    else:
        raise ValueError("pool_way not in ['max', 'mean']")
    f_act_hide = act_hide_dict[act_hide] if act_hide in list(range(1, 6)) else None
    ##########################################################################################
    cat_layer_list = []
    output_layer = input_layer = Input(shape=tuple([None for _ in range(nD)]) + tuple([X_channels]))

    for k in kernels_all[:-1]:
        cat_layer, output_layer = down_block(k)(output_layer)
        cat_layer_list.append(cat_layer)
    kernels_all.reverse()
    for k in kernels_all[:-1]:
        cat_layer = cat_layer_list.pop()
        output_layer = up_block(k, cat_layer)(output_layer)
    output_layer = out_block([kernels_all[-1], Y_channels])(output_layer)
    if final_minus:
        output_layer = subtract([input_layer, output_layer])
    return Model(inputs=input_layer, outputs=output_layer)


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    unet = FlexibleBridgeNet(4, kernels_all=[4, 8, 16, 32, 64, 128], final_minus=True, res_case=3, res_number=2)
    unet.summary(line_length=150)
