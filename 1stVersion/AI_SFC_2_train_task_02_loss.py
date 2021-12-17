import os
import numpy as np

loss_test = np.zeros(shape=(4, 1), dtype=np.int32)
loss_test[0, 0] = 1
loss_test[1, 0] = 2
loss_test[2, 0] = 3
loss_test[3, 0] = 4

for i in range(len(loss_test)):
    os.system(
        "python ./AI_SFC_2_train.py "
        "--gpuID 0 "
        "--training_data_disk G "
        "--models_disk G "
        "--stride 8 8 "
        "--loss_used " + str(loss_test[i, 0]) + " "
                                                "--pool_way 1 "
                                                "--act_hide 1 "
                                                "--kernel_size 11 "
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