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

tic
[TrainingData_Labels]=ReadSegyFast('TrainingData_Labels.segy');
toc
TrainingData_Labels = reshape(TrainingData_Labels,[Nt xlines_training ilines_training]);
tic
Training_Test1_Test2_Image_all = BinDataRead(['D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' 'Training_Test1_Test2_Image_1006x1116x841.bin'],[Nt,xlines_test1+xlines_test2,ilines_training+ilines_test1]);
toc
Training_Test1_Test2_Image_Pro = Training_Test1_Test2_Image_all(1:ZsPro, 1:XlinesPro,1:IlinesPro);
clear Training_Test1_Test2_Image_all;

eval_scores_ID = 1;
eval_scores_Best = BinDataRead('3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1_dila1_repro2_models3_3_cube_eval_scores_1x5x1.bin',[1 5 1]);
Best_iD = ['3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1_dila1_repro2_Y_pred_1006x1116x841_model3'];
tic
Training_Test1_Test2_Labels_Best = BinDataRead([Best_iD '.bin'],[ZsPro XlinesPro IlinesPro]);
toc

paper_plot_3D_best;
