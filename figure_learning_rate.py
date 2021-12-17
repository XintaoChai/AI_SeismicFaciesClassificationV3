from pyseismic_local.plot_functions import *
from pyseismic_local.data_read_write_functions import *
import pandas as pd
################################################################################
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
################################################################################
paper_figure_dir = './' + 'paper_figure'
if not os.path.exists(paper_figure_dir):
    os.mkdir(paper_figure_dir)

log_root_dir = 'G:/BigData/AI_SeismicFaciesClassification/models/'
################################################################################
epoch_max = 10

log_lr001 = pd.read_csv(log_root_dir + '2patchR992C576_batch16_number10000_stride8_8_loss5_lr10_pool1_kernelSize11_Net5_kernels16_repeat1_repro0' + '/log.csv', header=0)
log_lr010 = pd.read_csv(log_root_dir + '2patchR992C576_batch16_number10000_stride8_8_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1_repro0' + '/log.csv', header=0)
log_lr050 = pd.read_csv(log_root_dir + '2patchR992C576_batch16_number10000_stride8_8_loss5_lr500_pool1_kernelSize11_Net5_kernels16_repeat1_repro0' + '/log.csv', header=0)
log_lr100 = pd.read_csv(log_root_dir + '2patchR992C576_batch16_number10000_stride8_8_loss5_lr1000_pool1_kernelSize11_Net5_kernels16_repeat1_repro0' + '/log.csv', header=0)
log_lr500 = pd.read_csv(log_root_dir + '2patchR992C576_batch16_number10000_stride8_8_loss5_lr5000_pool1_kernelSize11_Net5_kernels16_repeat1_repro0' + '/log.csv', header=0)

paper_results_dir = 'D:/Nutstore/AI_ML_DL_WeDo/AI_SeismicFaciesClassification/paper_results/'

eval_scores_lr001_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr1_pool1_kernelSize11_Net5_kernels16_repeat1_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_lr001 = bin_data_reader3D_float32(eval_scores_lr001_ID, 1, 5, 1, 1, 1)
eval_scores_lr001 = np.reshape(eval_scores_lr001, (1, 5, 1))
print('eval_scores_lr001: ' + str(eval_scores_lr001))

eval_scores_lr010_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr10_pool1_kernelSize11_Net5_kernels16_repeat1_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_lr010 = bin_data_reader3D_float32(eval_scores_lr010_ID, 1, 5, 1, 1, 1)
eval_scores_lr010 = np.reshape(eval_scores_lr010, (1, 5, 1))
print('eval_scores_lr010: ' + str(eval_scores_lr010))

eval_scores_lr050_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr50_pool1_kernelSize11_Net5_kernels16_repeat1_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_lr050 = bin_data_reader3D_float32(eval_scores_lr050_ID, 1, 5, 1, 1, 1)
eval_scores_lr050 = np.reshape(eval_scores_lr050, (1, 5, 1))
print('eval_scores_lr050: ' + str(eval_scores_lr050))

eval_scores_lr100_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1_dila1_repro2_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_lr100 = bin_data_reader3D_float32(eval_scores_lr100_ID, 1, 5, 1, 1, 1)
eval_scores_lr100 = np.reshape(eval_scores_lr100, (1, 5, 1))
print('eval_scores_lr100: ' + str(eval_scores_lr100))

eval_scores_lr500_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr500_pool1_kernelSize11_Net5_kernels16_repeat1_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_lr500 = bin_data_reader3D_float32(eval_scores_lr500_ID, 1, 5, 1, 1, 1)
eval_scores_lr500 = np.reshape(eval_scores_lr500, (1, 5, 1))
print('eval_scores_lr500: ' + str(eval_scores_lr500))

eval_scores_SNR = np.zeros(shape=(5, 1), dtype='float32')
SNR_index = 0
eval_scores_SNR[0, 0] = eval_scores_lr001[0, SNR_index, 0]
eval_scores_SNR[1, 0] = eval_scores_lr010[0, SNR_index, 0]
eval_scores_SNR[2, 0] = eval_scores_lr050[0, SNR_index, 0]
eval_scores_SNR[3, 0] = eval_scores_lr100[0, SNR_index, 0]
eval_scores_SNR[4, 0] = eval_scores_lr500[0, SNR_index, 0]

eval_scores_accAllClasses = np.zeros(shape=(5, 1), dtype='float32')
accAllClasses_index = 1
eval_scores_accAllClasses[0, 0] = eval_scores_lr001[0, accAllClasses_index, 0]
eval_scores_accAllClasses[1, 0] = eval_scores_lr010[0, accAllClasses_index, 0]
eval_scores_accAllClasses[2, 0] = eval_scores_lr050[0, accAllClasses_index, 0]
eval_scores_accAllClasses[3, 0] = eval_scores_lr100[0, accAllClasses_index, 0]
eval_scores_accAllClasses[4, 0] = eval_scores_lr500[0, accAllClasses_index, 0]

eval_scores_accClassesAvg = np.zeros(shape=(5, 1), dtype='float32')
accClassesAvg_index = 2
eval_scores_accClassesAvg[0, 0] = eval_scores_lr001[0, accClassesAvg_index, 0]
eval_scores_accClassesAvg[1, 0] = eval_scores_lr010[0, accClassesAvg_index, 0]
eval_scores_accClassesAvg[2, 0] = eval_scores_lr050[0, accClassesAvg_index, 0]
eval_scores_accClassesAvg[3, 0] = eval_scores_lr100[0, accClassesAvg_index, 0]
eval_scores_accClassesAvg[4, 0] = eval_scores_lr500[0, accClassesAvg_index, 0]

eval_scores_mIoU = np.zeros(shape=(5, 1), dtype='float32')
mIoU_index = 3
eval_scores_mIoU[0, 0] = eval_scores_lr001[0, mIoU_index, 0]
eval_scores_mIoU[1, 0] = eval_scores_lr010[0, mIoU_index, 0]
eval_scores_mIoU[2, 0] = eval_scores_lr050[0, mIoU_index, 0]
eval_scores_mIoU[3, 0] = eval_scores_lr100[0, mIoU_index, 0]
eval_scores_mIoU[4, 0] = eval_scores_lr500[0, mIoU_index, 0]

eval_scores_wIoU = np.zeros(shape=(5, 1), dtype='float32')
wIoU_index = 4
eval_scores_wIoU[0, 0] = eval_scores_lr001[0, wIoU_index, 0]
eval_scores_wIoU[1, 0] = eval_scores_lr010[0, wIoU_index, 0]
eval_scores_wIoU[2, 0] = eval_scores_lr050[0, wIoU_index, 0]
eval_scores_wIoU[3, 0] = eval_scores_lr100[0, wIoU_index, 0]
eval_scores_wIoU[4, 0] = eval_scores_lr500[0, wIoU_index, 0]

figure_save = 1
dpi = 600

ylim_loss_train = (-1.48, 0.02)
ylim_accuracy_train = (0.42, 1.008)
ylim_accuracy_lr = (0.28, 1.02)
ticks_epoch = list(range(1, 11))
xlim_epoch = (1 - 0.2, 10 + 0.2)
xlim_lr = (0 - 0.1, 4 + 0.1)
epoch_axis = np.zeros(shape=(epoch_max, 1), dtype='int32')
epoch_axis[:, 0] = np.array(range(epoch_max)) + 1

color_lr001 = 'black'
marker_lr001 = 's'
linestyle_lr001 = 'solid'
linewidth_lr001 = 0.5
markersize_lr001 = 5

color_lr010 = 'magenta'
marker_lr010 = 'o'
linestyle_lr010 = linestyle_lr001
linewidth_lr010 = linewidth_lr001
markersize_lr010 = markersize_lr001

color_lr050 = 'green'
marker_lr050 = 'p'
linestyle_lr050 = linestyle_lr001
linewidth_lr050 = linewidth_lr001
markersize_lr050 = markersize_lr001

color_lr100 = 'blue'
marker_lr100 = '*'
linestyle_lr100 = linestyle_lr001
linewidth_lr100 = linewidth_lr001
markersize_lr100 = markersize_lr001

color_lr500 = 'red'
marker_lr500 = 'x'
linestyle_lr500 = linestyle_lr001
linewidth_lr500 = linewidth_lr001
markersize_lr500 = markersize_lr001

labelpad_epoch_loss = -0.5
labelpad_epoch_acc = labelpad_epoch_loss
labelpad_loss = 0
labelpad_acc = labelpad_loss
font_normal = {'family': 'Arial', 'weight': 'normal', 'size': 8}

axis_linewith = 0.3
fig_width = 3.33 * 1.12311
fig_height1 = 3.39
fig_height2 = 3.0
fig_height3 = 3.999
position_fig_h = 100
position_fig_v = 50
d_fig_h = 400
d_fig_v = 400
wm_geometry_1 = '+' + str(position_fig_h + 0 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_2 = '+' + str(position_fig_h + 1 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_3 = '+' + str(position_fig_h + 0 * d_fig_h) + '+' + str(position_fig_v + 1 * d_fig_v)

fig = plt.figure()
plt.rcParams['axes.unicode_minus'] = False
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_1)
fig.set_size_inches(fig_width, fig_height1)
plt.plot(epoch_axis, np.log10(log_lr001['loss'].values[0:epoch_max] / log_lr001['loss'].values[0]), color=color_lr001, linestyle=linestyle_lr001, linewidth=linewidth_lr001,
         marker=marker_lr001, markersize=markersize_lr001)
plt.plot(epoch_axis, np.log10(log_lr010['loss'].values[0:epoch_max] / log_lr010['loss'].values[0]), color=color_lr010, linestyle=linestyle_lr010, linewidth=linewidth_lr010,
         marker=marker_lr010, markersize=markersize_lr010)
plt.plot(epoch_axis, np.log10(log_lr050['loss'].values[0:epoch_max] / log_lr050['loss'].values[0]), color=color_lr050, linestyle=linestyle_lr050, linewidth=linewidth_lr050,
         marker=marker_lr050, markersize=markersize_lr050)
plt.plot(epoch_axis, np.log10(log_lr100['loss'].values[0:epoch_max] / log_lr100['loss'].values[0]), color=color_lr100, linestyle=linestyle_lr100, linewidth=linewidth_lr100,
         marker=marker_lr100, markersize=markersize_lr100)
plt.plot(epoch_axis, np.log10(log_lr500['loss'].values[0:epoch_max] / log_lr500['loss'].values[0]), color=color_lr500, linestyle=linestyle_lr500, linewidth=linewidth_lr500,
         marker=marker_lr500, markersize=markersize_lr500)
plt.legend(('$\mathit{lr}$ = 0.000001', '$\mathit{lr}$ = 0.000010', '$\mathit{lr}$ = 0.000050', '$\mathit{lr}$ = 0.000100', '$\mathit{lr}$ = 0.000500'), loc='best',
           fontsize=font_normal['size'], frameon=False, prop=font_normal)
plt.xlabel('Epoch', font=font_normal, labelpad=labelpad_epoch_loss)
plt.ylabel('log$_{10}$(normalized loss)', font=font_normal, labelpad=labelpad_loss)
plt.xticks(ticks_epoch, font=font_normal)
plt.yticks(np.linspace(-1.5, 0, num=int((0 - (-1.5)) / 0.1 + 1), endpoint=True), font=font_normal)
plt.xlim(xlim_epoch)
plt.ylim(ylim_loss_train)
fname = paper_figure_dir + '/' + 'FigTrainLossLR.tif'
if figure_save:
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

fig = plt.figure()
plt.rcParams['axes.unicode_minus'] = False
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_2)
fig.set_size_inches(fig_width, fig_height2)
plt.plot(epoch_axis, log_lr001['accuracy'].values[0:epoch_max], color=color_lr001, linestyle=linestyle_lr001, linewidth=linewidth_lr001, marker=marker_lr001,
         markersize=markersize_lr001)
plt.plot(epoch_axis, log_lr010['accuracy'].values[0:epoch_max], color=color_lr010, linestyle=linestyle_lr010, linewidth=linewidth_lr010, marker=marker_lr010,
         markersize=markersize_lr010)
plt.plot(epoch_axis, log_lr050['accuracy'].values[0:epoch_max], color=color_lr050, linestyle=linestyle_lr050, linewidth=linewidth_lr050, marker=marker_lr050,
         markersize=markersize_lr050)
plt.plot(epoch_axis, log_lr100['accuracy'].values[0:epoch_max], color=color_lr100, linestyle=linestyle_lr100, linewidth=linewidth_lr100, marker=marker_lr100,
         markersize=markersize_lr100)
plt.plot(epoch_axis, log_lr500['accuracy'].values[0:epoch_max], color=color_lr500, linestyle=linestyle_lr500, linewidth=linewidth_lr500, marker=marker_lr500,
         markersize=markersize_lr500)
plt.legend(('$\mathit{lr}$ = 0.000001', '$\mathit{lr}$ = 0.000010', '$\mathit{lr}$ = 0.000050', '$\mathit{lr}$ = 0.000100', '$\mathit{lr}$ = 0.000500'), loc='best',
           fontsize=font_normal['size'], frameon=False, prop=font_normal)
plt.xlabel('Epoch', font=font_normal, labelpad=labelpad_epoch_acc)
plt.ylabel('Accuracy', font=font_normal, labelpad=labelpad_acc)
plt.xticks(ticks_epoch, font=font_normal)
plt.yticks(np.linspace(0, 1, num=int((1 - 0) / 0.05 + 1), endpoint=True), font=font_normal)
plt.xlim(xlim_epoch)
plt.ylim(ylim_accuracy_train)
fname = paper_figure_dir + '/' + 'FigTrainAccLR.tif'
if figure_save:
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

itick = [0, 1, 2, 3, 4]
iticklabel = ['$\mathit{lr}$ = 0.000001', '$\mathit{lr}$ = 0.000010', '$\mathit{lr}$ = 0.000050', '$\mathit{lr}$ = 0.000100', '$\mathit{lr}$ = 0.000500']
fig = plt.figure()
ax = fig.add_subplot(211)
plt.rcParams['axes.unicode_minus'] = False
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_3)
fig.set_size_inches(fig_width, fig_height3)
plt.plot(eval_scores_accAllClasses, color=color_lr001, linestyle=linestyle_lr001, linewidth=linewidth_lr001, marker=marker_lr001, markersize=markersize_lr001)
plt.plot(eval_scores_accClassesAvg, color=color_lr010, linestyle=linestyle_lr010, linewidth=linewidth_lr010, marker=marker_lr010, markersize=markersize_lr010)
plt.plot(eval_scores_mIoU, color=color_lr050, linestyle=linestyle_lr050, linewidth=linewidth_lr050, marker=marker_lr050, markersize=markersize_lr050)
plt.plot(eval_scores_wIoU, color=color_lr100, linestyle=linestyle_lr100, linewidth=linewidth_lr100, marker=marker_lr100, markersize=markersize_lr100)
ax.xaxis.set_ticks(itick)
ax.xaxis.set_ticklabels(iticklabel, fontsize=font_normal['size'], color='black', family=font_normal['family'])
for label in ax.xaxis.get_ticklabels():
    label.set_ha('right')
    label.set_rotation(61)
    label.set_rotation_mode('default')  # {None, 'default', 'anchor'} 可注释掉
    label.set_fontsize(font_normal['size'])
    label.set_y(0.05)  # 上下移动
plt.legend(('Accuracy over all classes', 'Average accuracy', 'Average IoU', 'Weighted IoU'), loc='best', fontsize=font_normal['size'], frameon=False, prop=font_normal)
plt.ylabel('Accuracy', font=font_normal, labelpad=labelpad_acc)
plt.xticks(font=font_normal)
plt.yticks(np.linspace(0, 1, num=int((1 - 0) / 0.05 + 1), endpoint=True), font=font_normal)
plt.xlim(xlim_lr)
plt.ylim(ylim_accuracy_lr)
fname = paper_figure_dir + '/' + 'FigEvalScoresLR.tif'
if figure_save:
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

plt.show()
