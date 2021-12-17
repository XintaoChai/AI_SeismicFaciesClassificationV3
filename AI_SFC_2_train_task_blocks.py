import os
import numpy as np

repeat_blocks = np.zeros(shape=(2, 1), dtype=np.int32)
repeat_blocks[0, 0] = 0
repeat_blocks[1, 0] = 2

for i in range(len(repeat_blocks)):
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
        "--BridgeNet_used 5 "
        "--training_number 10000 "
        "--epochs 5 "
        "--lr 0.0001 "
        "--kernels_all 16 32 64 128 256 512 "
        "--batch_size 16 "
        "--patch_rows 992 "
        "--patch_cols 576 "
        "--conv2act_repeat " + str(repeat_blocks[i, 0]) + " "
                                                          "--reproduce 0 "
                                                          "--save_every 1 "
        # "--plot_show "
        # "--continue_train "
    )
