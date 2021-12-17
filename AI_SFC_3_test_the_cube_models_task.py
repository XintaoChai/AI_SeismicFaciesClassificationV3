import os

model_folder_ID = []
# model_folder_ID.append('patchR0992C0576_batch016_number10000_stride8_8_lr1000_kSize11_ResCase0_ResNo1_Net5_Crep2_Repro01_kernels16_32_64_128_256_512')
#
# model_folder_ID.append('patchR0992C0576_batch016_number10000_stride8_8_lr1000_kSize11_ResCase3_ResNo1_Net5_Crep2_Repro01_kernels16_32_64_128_256_512')
# model_folder_ID.append('patchR0992C0576_batch016_number10000_stride8_8_lr1000_kSize11_ResCase3_ResNo1_Net5_Crep2_Repro02_kernels16_32_64_128_256_512')
# model_folder_ID.append('patchR0992C0576_batch016_number10000_stride8_8_lr1000_kSize11_ResCase3_ResNo1_Net5_Crep2_Repro03_kernels16_32_64_128_256_512')
model_folder_ID.append('patchR0992C0576_batch016_number10000_stride8_8_lr1000_kSize11_ResCase3_ResNo1_Net5_Crep2_Repro04_kernels16_32_64_128_256_512')
model_folder_ID.append('patchR0992C0576_batch016_number10000_stride8_8_lr1000_kSize11_ResCase3_ResNo1_Net5_Crep2_Repro05_kernels16_32_64_128_256_512')
model_folder_ID.append('patchR0992C0576_batch016_number10000_stride8_8_lr1000_kSize11_ResCase3_ResNo1_Net5_Crep2_Repro06_kernels16_32_64_128_256_512')
for i_model in range(len(model_folder_ID)):
    os.system(
        "python ./AI_SFC_3_test_the_cube_models.py "
        "--gpuID 0 "
        "--iline_way 1 "
        "--model_folder_id " + model_folder_ID[i_model] + " "
                                                          "--Y_1 1 "
                                                          "--model_number_begin 2 "
                                                          "--model_number_end 2 "
        # "--plot_show "
    )
