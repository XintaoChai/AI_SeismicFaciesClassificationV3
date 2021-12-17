import os
import numpy as np

depth_sets = np.zeros(shape=(4, 1), dtype=np.int32)
depth_sets[0, 0] = 4
depth_sets[1, 0] = 3
depth_sets[2, 0] = 2
depth_sets[3, 0] = 1

for i in range(len(depth_sets)):
    os.system(
        "python ./AI_SFC_2_train.py "
        "--gpuID 0 "
        "--training_data_disk G "
        "--models_disk G "
        "--stride 8 8 "
        "--loss_used 5 "
        "--pool_way 1 "
        "--act_hide 1 "
        "--kernel_size 11 "
        "--BridgeNet_used " + str(depth_sets[i, 0]) + " "
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
