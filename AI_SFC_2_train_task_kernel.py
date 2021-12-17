import os
import numpy as np

kernel_size = np.zeros(shape=(9, 1), dtype=np.int32)
kernel_size[0, 0] = 2
kernel_size[1, 0] = 3
kernel_size[2, 0] = 4
kernel_size[3, 0] = 5
kernel_size[4, 0] = 6
kernel_size[5, 0] = 7
kernel_size[6, 0] = 8
kernel_size[7, 0] = 9
kernel_size[8, 0] = 10

for i in range(len(kernel_size)):
    os.system(
        "python ./AI_SFC_2_train.py "
        "--gpuID 0 "
        "--training_data_disk G "
        "--models_disk G "
        "--stride 8 8 "
        "--loss_used 5 "
        "--pool_way 1 "
        "--act_hide 1 "
        "--kernel_size " + str(kernel_size[i, 0]) + " "
                                                    "--BridgeNet_used 5 "
                                                    "--training_number 10000 "
                                                    "--epochs 5 "
                                                    "--lr 0.0001 "
                                                    "--kernels_all 16 32 64 128 256 512 "
                                                    "--batch_size 16 "
                                                    "--patch_rows 992 "
                                                    "--patch_cols 576 "
                                                    "--conv2act_repeat 1 "
                                                    "--reproduce 0 "
                                                    "--save_every 1 "
        # "--plot_show "
        # "--continue_train "
    )
