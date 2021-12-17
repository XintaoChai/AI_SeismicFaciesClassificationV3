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

log_root_dir = 'D:/BigData/AI_SeismicFaciesClassification/models/'
################################################################################
epoch_max = 3

log_kernel03 = pd.read_csv(log_root_dir + 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize3_Net5_kernels16_repeat1_dila1_repro1_norm5522' + '/log.csv', header=0)
log_kernel05 = pd.read_csv(log_root_dir + 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize5_Net5_kernels16_repeat1_dila1_repro1_norm5522' + '/log.csv', header=0)
log_kernel07 = pd.read_csv(log_root_dir + 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize7_Net5_kernels16_repeat1_dila1_repro1_norm5522' + '/log.csv', header=0)
log_kernel09 = pd.read_csv(log_root_dir + 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize9_Net5_kernels16_repeat1_dila1_repro1_norm5522' + '/log.csv', header=0)
log_kernel11 = pd.read_csv(log_root_dir + 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1_dila1_repro2' + '/log.csv', header=0)

paper_results_dir = 'D:/Nutstore/AI_ML_DL_WeDo/AI_SeismicFaciesClassification/paper_results/'

eval_scores_kernel03_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize3_Net5_kernels16_repeat1_dila1_repro1_norm5522_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_kernel03 = bin_data_reader3D_float32(eval_scores_kernel03_ID, 1, 5, 1, 1, 1)
eval_scores_kernel03 = np.reshape(eval_scores_kernel03, (1, 5, 1))
print('eval_scores_kernel03: ' + str(eval_scores_kernel03))

eval_scores_kernel05_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize5_Net5_kernels16_repeat1_dila1_repro1_norm5522_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_kernel05 = bin_data_reader3D_float32(eval_scores_kernel05_ID, 1, 5, 1, 1, 1)
eval_scores_kernel05 = np.reshape(eval_scores_kernel05, (1, 5, 1))
print('eval_scores_kernel05: ' + str(eval_scores_kernel05))

eval_scores_kernel07_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize7_Net5_kernels16_repeat1_dila1_repro1_norm5522_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_kernel07 = bin_data_reader3D_float32(eval_scores_kernel07_ID, 1, 5, 1, 1, 1)
eval_scores_kernel07 = np.reshape(eval_scores_kernel07, (1, 5, 1))
print('eval_scores_kernel07: ' + str(eval_scores_kernel07))

eval_scores_kernel09_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize9_Net5_kernels16_repeat1_dila1_repro1_norm5522_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_kernel09 = bin_data_reader3D_float32(eval_scores_kernel09_ID, 1, 5, 1, 1, 1)
eval_scores_kernel09 = np.reshape(eval_scores_kernel09, (1, 5, 1))
print('eval_scores_kernel09: ' + str(eval_scores_kernel09))

eval_scores_kernel11_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1_dila1_repro2_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_kernel11 = bin_data_reader3D_float32(eval_scores_kernel11_ID, 1, 5, 1, 1, 1)
eval_scores_kernel11 = np.reshape(eval_scores_kernel11, (1, 5, 1))
print('eval_scores_kernel11: ' + str(eval_scores_kernel11))

eval_scores_SNR = np.zeros(shape=(5, 1), dtype='float32')
SNR_index = 0
eval_scores_SNR[0, 0] = eval_scores_kernel03[0, SNR_index, 0]
eval_scores_SNR[1, 0] = eval_scores_kernel05[0, SNR_index, 0]
eval_scores_SNR[2, 0] = eval_scores_kernel07[0, SNR_index, 0]
eval_scores_SNR[3, 0] = eval_scores_kernel09[0, SNR_index, 0]
eval_scores_SNR[4, 0] = eval_scores_kernel11[0, SNR_index, 0]

eval_scores_accAllClasses = np.zeros(shape=(5, 1), dtype='float32')
accAllClasses_index = 1
eval_scores_accAllClasses[0, 0] = eval_scores_kernel03[0, accAllClasses_index, 0]
eval_scores_accAllClasses[1, 0] = eval_scores_kernel05[0, accAllClasses_index, 0]
eval_scores_accAllClasses[2, 0] = eval_scores_kernel07[0, accAllClasses_index, 0]
eval_scores_accAllClasses[3, 0] = eval_scores_kernel09[0, accAllClasses_index, 0]
eval_scores_accAllClasses[4, 0] = eval_scores_kernel11[0, accAllClasses_index, 0]

eval_scores_accClassesAvg = np.zeros(shape=(5, 1), dtype='float32')
accClassesAvg_index = 2
eval_scores_accClassesAvg[0, 0] = eval_scores_kernel03[0, accClassesAvg_index, 0]
eval_scores_accClassesAvg[1, 0] = eval_scores_kernel05[0, accClassesAvg_index, 0]
eval_scores_accClassesAvg[2, 0] = eval_scores_kernel07[0, accClassesAvg_index, 0]
eval_scores_accClassesAvg[3, 0] = eval_scores_kernel09[0, accClassesAvg_index, 0]
eval_scores_accClassesAvg[4, 0] = eval_scores_kernel11[0, accClassesAvg_index, 0]

eval_scores_mIoU = np.zeros(shape=(5, 1), dtype='float32')
mIoU_index = 3
eval_scores_mIoU[0, 0] = eval_scores_kernel03[0, mIoU_index, 0]
eval_scores_mIoU[1, 0] = eval_scores_kernel05[0, mIoU_index, 0]
eval_scores_mIoU[2, 0] = eval_scores_kernel07[0, mIoU_index, 0]
eval_scores_mIoU[3, 0] = eval_scores_kernel09[0, mIoU_index, 0]
eval_scores_mIoU[4, 0] = eval_scores_kernel11[0, mIoU_index, 0]

eval_scores_wIoU = np.zeros(shape=(5, 1), dtype='float32')
wIoU_index = 4
eval_scores_wIoU[0, 0] = eval_scores_kernel03[0, wIoU_index, 0]
eval_scores_wIoU[1, 0] = eval_scores_kernel05[0, wIoU_index, 0]
eval_scores_wIoU[2, 0] = eval_scores_kernel07[0, wIoU_index, 0]
eval_scores_wIoU[3, 0] = eval_scores_kernel09[0, wIoU_index, 0]
eval_scores_wIoU[4, 0] = eval_scores_kernel11[0, wIoU_index, 0]

figure_save = 1
dpi = 300

ylim_loss_train = (-1.09, 0.02)
ylim_accuracy_train = (0.73, 1.003999)
ylim_accuracy_kernel = (0.805, 1.006)
ticks_epoch = [1, 2, 3]
xlim_epoch = (1 - 0.05, 3 + 0.05)
xlim_kernel = (0 - 0.1, 4 + 0.1)
epoch_axis = np.zeros(shape=(epoch_max, 1), dtype='int32')
epoch_axis[:, 0] = np.array(range(epoch_max)) + 1

color_kernel03 = 'black'
marker_kernel03 = 's'
linestyle_kernel03 = '-'
linewidth_kernel03 = 0.5
markersize_kernel03 = 5

color_kernel05 = 'magenta'
marker_kernel05 = 'o'
linestyle_kernel05 = '-'
linewidth_kernel05 = linewidth_kernel03
markersize_kernel05 = markersize_kernel03

color_kernel07 = 'blue'
marker_kernel07 = '*'
linestyle_kernel07 = '-'
linewidth_kernel07 = linewidth_kernel03
markersize_kernel07 = markersize_kernel03

color_kernel09 = 'green'
marker_kernel09 = 'p'
linestyle_kernel09 = '-'
linewidth_kernel09 = linewidth_kernel03
markersize_kernel09 = markersize_kernel03

color_kernel11 = 'red'
marker_kernel11 = 'x'
linestyle_kernel11 = '-'
linewidth_kernel11 = linewidth_kernel03
markersize_kernel11 = markersize_kernel03

labelpad_epoch_loss = -1
labelpad_epoch_acc = labelpad_epoch_loss
labelpad_loss = 0
labelpad_acc = -1.1
font_normal = {'family': 'Arial', 'weight': 'normal', 'size': 8}

axis_linewith = 0.3
fig_width = 3.33 * 1.12311
fig_height1 = 1.6
fig_height2 = 3.46
fig_height3 = 4.9
position_fig_h = 100
position_fig_v = 50
d_fig_h = 400
d_fig_v = 400
wm_geometry_1 = '+' + str(position_fig_h + 0 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_2 = '+' + str(position_fig_h + 1 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_3 = '+' + str(position_fig_h + 0 * d_fig_h) + '+' + str(position_fig_v + 1 * d_fig_v)

itick = [0, 1, 2, 3, 4]
iticklabel = ['Kernel size: 3 x 3', 'Kernel size: 5 x 5', 'Kernel size: 7 x 7', 'Kernel size: 9 x 9', 'Kernel size: 11 x 11']
fig = plt.figure()
ax = fig.add_subplot(211)
plt.rcParams['axes.unicode_minus'] = False
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_3)
fig.set_size_inches(fig_width, fig_height3)
plt.plot(eval_scores_accAllClasses, color=color_kernel03, linestyle=linestyle_kernel03, linewidth=linewidth_kernel03, marker=marker_kernel03, markersize=markersize_kernel03)
plt.plot(eval_scores_accClassesAvg, color=color_kernel05, linestyle=linestyle_kernel05, linewidth=linewidth_kernel05, marker=marker_kernel05, markersize=markersize_kernel05)
plt.plot(eval_scores_mIoU, color=color_kernel07, linestyle=linestyle_kernel07, linewidth=linewidth_kernel07, marker=marker_kernel07, markersize=markersize_kernel07)
plt.plot(eval_scores_wIoU, color=color_kernel09, linestyle=linestyle_kernel09, linewidth=linewidth_kernel09, marker=marker_kernel09, markersize=markersize_kernel09)
ax.xaxis.set_ticks(itick)
ax.xaxis.set_ticklabels(iticklabel, fontsize=font_normal['size'], color='black', family=font_normal['family'])
for label in ax.xaxis.get_ticklabels():
    label.set_ha('right')
    label.set_rotation(65)
    label.set_rotation_mode('default')  # {None, 'default', 'anchor'} 可注释掉
    label.set_fontsize(font_normal['size'])
    label.set_y(0.04)  # 上下移动
plt.legend(('Accuracy over all classes', 'Average accuracy', 'Average IoU', 'Weighted IoU'), loc='best', fontsize=font_normal['size'], frameon=False, prop=font_normal)
plt.ylabel('Accuracy', font=font_normal, labelpad=labelpad_acc)
plt.xticks(font=font_normal)
plt.yticks(np.linspace(0, 1, num=int((1 - 0) / 0.02 + 1), endpoint=True), font=font_normal)
plt.xlim(xlim_kernel)
plt.ylim(ylim_accuracy_kernel)
fname = paper_figure_dir + '/' + 'FigEvalScoresKernel.tif'
if figure_save:
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

plt.show()
