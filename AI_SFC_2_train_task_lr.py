import os

lr_test = []
lr_test.append(0.0005000)
lr_test.append(0.0001000)
lr_test.append(0.0000500)
lr_test.append(0.0000100)
lr_test.append(0.0000010)

for i in range(len(lr_test)):
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
        "--epochs 10 "
        "--lr " + str(lr_test[i]) + " "
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
