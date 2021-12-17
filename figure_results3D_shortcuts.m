clc;clear;close all;
addpath(genpath('D:/BigData'));
addpath(genpath('G:/BigData'));
addpath(genpath('D:/Nutstore/EarthLab_Codes'));
addpath(genpath('D:/Nutstore/EarthLab_DataSmall'));
addpath(genpath(pwd));
opts.figures_path = './paper_figure/';
Nt = 1006;
dt = 0.003;
ilines_training = 590;
xlines_training = 782;

ilines_test1 = 251;
xlines_test1 = 782;

ilines_test2 = ilines_training+ilines_test1;
xlines_test2 = 334;

ZsPro = Nt;
XlinesPro = xlines_test1+xlines_test2;
IlinesPro = ilines_training+ilines_test1;

X_max = 5522.086;
X_min = -5195.5234;

iCaxisX = [X_min X_max]*0.1;
iCaxisY = [1 6];

iline_axis(:,1) = 1:IlinesPro;
xline_axis(:,1) = 1:XlinesPro;
time_axis(:,1) = ((1:ZsPro)-1)*dt;

red_rgb = [255 0 0];
yellow_rgb = [255 255 0];
green_rgb = [0 128 0];
blue_rgb = [0 0 255];
fuchsia_rgb=[255 0 255];
brown_rgb=[165 42 42];
color_6 = [red_rgb
    yellow_rgb
    green_rgb
    blue_rgb
    fuchsia_rgb
    brown_rgb];
iColormapY = color_6/255;


eval_scores_ID = 1;
net_case = 0;
eval_scores_ResNet4 = BinDataRead('3_All_iline_patchR0992C0576_batch016_number10000_stride8_8_lr1000_kSize11_ResCase0_ResNo1_Net5_Crep2_Repro01_kernels16_32_64_128_256_512_models2_2_cube_eval_scores_1x5x1.bin',[1 5 1]);
ResNet4_iD = ['3_All_iline_patchR0992C0576_batch016_number10000_stride8_8_lr1000_kSize11_ResCase0_ResNo1_Net5_Crep2_Repro01_kernels16_32_64_128_256_512_Y_pred_1006x1116x841_model2'];

tic
Training_Test1_Test2_Labels_ResNet4 = BinDataRead([ResNet4_iD '.bin'],[ZsPro XlinesPro IlinesPro]);
toc

eval_scores_ID = 1;
eval_scores_Best = BinDataRead('3_All_iline_patchR0992C0576_batch016_number10000_stride8_8_lr1000_kSize11_ResCase3_ResNo1_Net5_Crep2_Repro01_kernels16_32_64_128_256_512_models2_2_cube_eval_scores_1x5x1.bin',[1 5 1]);
Best_iD = ['3_All_iline_patchR0992C0576_batch016_number10000_stride8_8_lr1000_kSize11_ResCase3_ResNo1_Net5_Crep2_Repro01_kernels16_32_64_128_256_512_Y_pred_1006x1116x841_model2'];
tic
Training_Test1_Test2_Labels_Best = BinDataRead([Best_iD '.bin'],[ZsPro XlinesPro IlinesPro]);
toc

paper_plot_3D_ResNet;


