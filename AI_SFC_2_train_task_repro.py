import os

for i in range(100):
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
        "--lr 0.0001000 "
        "--kernels_all 16 32 64 128 256 512 "
        "--batch_size 16 "
        "--patch_rows 992 "
        "--patch_cols 576 "
        "--conv2act_repeat 1 "
        "--reproduce " + str(i + 3) + " "
                                      "--save_every 1 "
        # "--plot_show "
        # "--continue_train "
    )
