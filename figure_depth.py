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

log_depth1 = pd.read_csv(log_root_dir + 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net1_kernels16_repeat1_dila1_repro1_norm5522' + '/log.csv', header=0)
log_depth2 = pd.read_csv(log_root_dir + 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net2_kernels16_repeat1_dila1_repro1_norm5522' + '/log.csv', header=0)
log_depth3 = pd.read_csv(log_root_dir + 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net3_kernels16_repeat1_dila1_repro1_norm5522' + '/log.csv', header=0)
log_depth4 = pd.read_csv(log_root_dir + 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net4_kernels16_repeat1_dila1_repro1_norm5522' + '/log.csv', header=0)
log_depth5 = pd.read_csv(log_root_dir + 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1_dila1_repro2' + '/log.csv', header=0)

paper_results_dir = './paper_results/'

eval_scores_depth1_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net1_kernels16_repeat1_dila1_repro1_norm5522_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_depth1 = bin_data_reader3D_float32(eval_scores_depth1_ID, 1, 5, 1, 1, 1)
eval_scores_depth1 = np.reshape(eval_scores_depth1, (1, 5, 1))
print('eval_scores_depth1: ' + str(eval_scores_depth1))

eval_scores_depth2_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net2_kernels16_repeat1_dila1_repro1_norm5522_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_depth2 = bin_data_reader3D_float32(eval_scores_depth2_ID, 1, 5, 1, 1, 1)
eval_scores_depth2 = np.reshape(eval_scores_depth2, (1, 5, 1))
print('eval_scores_depth2: ' + str(eval_scores_depth2))

eval_scores_depth3_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net3_kernels16_repeat1_dila1_repro1_norm5522_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_depth3 = bin_data_reader3D_float32(eval_scores_depth3_ID, 1, 5, 1, 1, 1)
eval_scores_depth3 = np.reshape(eval_scores_depth3, (1, 5, 1))
print('eval_scores_depth3: ' + str(eval_scores_depth3))

eval_scores_depth4_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net4_kernels16_repeat1_dila1_repro1_norm5522_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_depth4 = bin_data_reader3D_float32(eval_scores_depth4_ID, 1, 5, 1, 1, 1)
eval_scores_depth4 = np.reshape(eval_scores_depth4, (1, 5, 1))
print('eval_scores_depth4: ' + str(eval_scores_depth4))

eval_scores_depth5_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1_dila1_repro2_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_depth5 = bin_data_reader3D_float32(eval_scores_depth5_ID, 1, 5, 1, 1, 1)
eval_scores_depth5 = np.reshape(eval_scores_depth5, (1, 5, 1))
print('eval_scores_depth5: ' + str(eval_scores_depth5))

figure_save = 1
dpi = 300

ylim_loss_train = (-1.18, 0.02)
ylim_accuracy_train = (0.586, 1.006)
ylim_accuracy_model = (0.51, 1.010)
ticks_epoch = [1, 2, 3]
ticks_model = [1, 2, 3, 4, 5]
xlim_epoch = (1 - 0.05, 3 + 0.05)
xlim_model = (1 - 0.09, 5 + 0.09)
epoch_axis = np.zeros(shape=(epoch_max, 1), dtype='int32')
epoch_axis[:, 0] = np.array(range(epoch_max)) + 1
model_axis = np.zeros(shape=(5, 1), dtype='int32')
model_axis[:, 0] = np.array(range(5)) + 1

eval_scores_accAllClasses = np.zeros(shape=(5, 1), dtype='float32')
accAllClasses_index = 1
eval_scores_accAllClasses[0, 0] = eval_scores_depth1[0, accAllClasses_index, 0]
eval_scores_accAllClasses[1, 0] = eval_scores_depth2[0, accAllClasses_index, 0]
eval_scores_accAllClasses[2, 0] = eval_scores_depth3[0, accAllClasses_index, 0]
eval_scores_accAllClasses[3, 0] = eval_scores_depth4[0, accAllClasses_index, 0]
eval_scores_accAllClasses[4, 0] = eval_scores_depth5[0, accAllClasses_index, 0]

eval_scores_accClassesAvg = np.zeros(shape=(5, 1), dtype='float32')
accClassesAvg_index = 2
eval_scores_accClassesAvg[0, 0] = eval_scores_depth1[0, accClassesAvg_index, 0]
eval_scores_accClassesAvg[1, 0] = eval_scores_depth2[0, accClassesAvg_index, 0]
eval_scores_accClassesAvg[2, 0] = eval_scores_depth3[0, accClassesAvg_index, 0]
eval_scores_accClassesAvg[3, 0] = eval_scores_depth4[0, accClassesAvg_index, 0]
eval_scores_accClassesAvg[4, 0] = eval_scores_depth5[0, accClassesAvg_index, 0]

eval_scores_mIoU = np.zeros(shape=(5, 1), dtype='float32')
mIoU_index = 3
eval_scores_mIoU[0, 0] = eval_scores_depth1[0, mIoU_index, 0]
eval_scores_mIoU[1, 0] = eval_scores_depth2[0, mIoU_index, 0]
eval_scores_mIoU[2, 0] = eval_scores_depth3[0, mIoU_index, 0]
eval_scores_mIoU[3, 0] = eval_scores_depth4[0, mIoU_index, 0]
eval_scores_mIoU[4, 0] = eval_scores_depth5[0, mIoU_index, 0]

eval_scores_wIoU = np.zeros(shape=(5, 1), dtype='float32')
wIoU_index = 4
eval_scores_wIoU[0, 0] = eval_scores_depth1[0, wIoU_index, 0]
eval_scores_wIoU[1, 0] = eval_scores_depth2[0, wIoU_index, 0]
eval_scores_wIoU[2, 0] = eval_scores_depth3[0, wIoU_index, 0]
eval_scores_wIoU[3, 0] = eval_scores_depth4[0, wIoU_index, 0]
eval_scores_wIoU[4, 0] = eval_scores_depth5[0, wIoU_index, 0]

color_depth1 = 'black'
marker_depth1 = 's'
linestyle_depth1 = '-'
linewidth_depth1 = 0.5
markersize_depth1 = 5

color_depth2 = 'magenta'
marker_depth2 = 'o'
linestyle_depth2 = linestyle_depth1
linewidth_depth2 = linewidth_depth1
markersize_depth2 = markersize_depth1

color_depth3 = 'blue'
marker_depth3 = '*'
linestyle_depth3 = linestyle_depth1
linewidth_depth3 = linewidth_depth1
markersize_depth3 = markersize_depth1

color_depth4 = 'green'
marker_depth4 = 'p'
linestyle_depth4 = linestyle_depth1
linewidth_depth4 = linewidth_depth1
markersize_depth4 = markersize_depth1

color_depth5 = 'red'
marker_depth5 = 'x'
linestyle_depth5 = linestyle_depth1
linewidth_depth5 = linewidth_depth1
markersize_depth5 = markersize_depth1

labelpad_epoch_loss = -1
labelpad_epoch_acc = labelpad_epoch_loss
labelpad_loss = 0
labelpad_acc = -1.0
font_normal = {'family': 'Arial', 'weight': 'normal', 'size': 8}

axis_linewith = 0.3
fig_width = 3.33 * 1.12311
fig_height1 = 3.7
fig_height2 = fig_height1
fig_height3 = 7.2
position_fig_h = 100
position_fig_v = 50
d_fig_h = 400
d_fig_v = 400
wm_geometry_1 = '+' + str(position_fig_h + 0 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_2 = '+' + str(position_fig_h + 1 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_3 = '+' + str(position_fig_h + 0 * d_fig_h) + '+' + str(position_fig_v + 1 * d_fig_v)

# fig = plt.figure()
# plt.rcParams['axes.unicode_minus'] = False
# mngr = plt.get_current_fig_manager()
# mngr.window.wm_geometry(wm_geometry_1)
# fig.set_size_inches(fig_width, fig_height1)
# plt.plot(epoch_axis, np.log10(log_depth1['loss'].values[0:epoch_max] / log_depth1['loss'].values[0]), color=color_depth1, linestyle=linestyle_depth1, linewidth=linewidth_depth1, marker=marker_depth1,
#          markersize=markersize_depth1)
# plt.plot(epoch_axis, np.log10(log_depth2['loss'].values[0:epoch_max] / log_depth2['loss'].values[0]), color=color_depth2, linestyle=linestyle_depth2, linewidth=linewidth_depth2, marker=marker_depth2,
#          markersize=markersize_depth2)
# plt.plot(epoch_axis, np.log10(log_depth3['loss'].values[0:epoch_max] / log_depth3['loss'].values[0]), color=color_depth3, linestyle=linestyle_depth3, linewidth=linewidth_depth3, marker=marker_depth3,
#          markersize=markersize_depth3)
# plt.plot(epoch_axis, np.log10(log_depth4['loss'].values[0:epoch_max] / log_depth4['loss'].values[0]), color=color_depth4, linestyle=linestyle_depth4, linewidth=linewidth_depth4, marker=marker_depth4,
#          markersize=markersize_depth4)
# plt.plot(epoch_axis, np.log10(log_depth5['loss'].values[0:epoch_max] / log_depth5['loss'].values[0]), color=color_depth5, linestyle=linestyle_depth5, linewidth=linewidth_depth5, marker=marker_depth5,
#          markersize=markersize_depth5)
# plt.legend(('Pooling 1 time', 'Pooling 2 times', 'Pooling 3 times', 'Pooling 4 times', 'Pooling 5 times'), loc='best', fontsize=font_normal['size'], frameon=False)
# plt.xlabel('Epoch', font=font_normal, labelpad=labelpad_epoch_loss)
# plt.ylabel('Loss (normalized)', font=font_normal, labelpad=labelpad_loss)
# plt.xticks(ticks_epoch, font=font_normal)
# plt.yticks(np.linspace(-1.1, 0, num=int((0 - (-1.1)) / 0.1 + 1), endpoint=True), font=font_normal)
# plt.xlim(xlim_epoch)
# plt.ylim(ylim_loss_train)
# fname = paper_figure_dir + '/' + 'FigTrainLossDepth.tif'
# if figure_save:
#     plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')
#
# fig = plt.figure()
# plt.rcParams['axes.unicode_minus'] = False
# mngr = plt.get_current_fig_manager()
# mngr.window.wm_geometry(wm_geometry_2)
# fig.set_size_inches(fig_width, fig_height2)
# plt.plot(epoch_axis, log_depth1['accuracy'].values[0:epoch_max], color=color_depth1, linestyle=linestyle_depth1, linewidth=linewidth_depth1, marker=marker_depth1, markersize=markersize_depth1)
# plt.plot(epoch_axis, log_depth2['accuracy'].values[0:epoch_max], color=color_depth2, linestyle=linestyle_depth2, linewidth=linewidth_depth2, marker=marker_depth2, markersize=markersize_depth2)
# plt.plot(epoch_axis, log_depth3['accuracy'].values[0:epoch_max], color=color_depth3, linestyle=linestyle_depth3, linewidth=linewidth_depth3, marker=marker_depth3, markersize=markersize_depth3)
# plt.plot(epoch_axis, log_depth4['accuracy'].values[0:epoch_max], color=color_depth4, linestyle=linestyle_depth4, linewidth=linewidth_depth4, marker=marker_depth4, markersize=markersize_depth4)
# plt.plot(epoch_axis, log_depth5['accuracy'].values[0:epoch_max], color=color_depth5, linestyle=linestyle_depth5, linewidth=linewidth_depth5, marker=marker_depth5, markersize=markersize_depth5)
# plt.legend(('Pooling 1 time', 'Pooling 2 times', 'Pooling 3 times', 'Pooling 4 times', 'Pooling 5 times'), loc='best', fontsize=font_normal['size'], frameon=False)
# plt.xlabel('Epoch', font=font_normal, labelpad=labelpad_epoch_acc)
# plt.ylabel('Accuracy', font=font_normal, labelpad=labelpad_acc)
# plt.xticks(ticks_epoch, font=font_normal)
# plt.yticks(np.linspace(0, 1, num=int((1 - 0) / 0.02 + 1), endpoint=True), font=font_normal)
# plt.xlim(xlim_epoch)
# plt.ylim(ylim_accuracy_train)
# fname = paper_figure_dir + '/' + 'FigTrainAccDepth.tif'
# if figure_save:
#     plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

itick = [1, 2, 3, 4, 5]
iticklabel = ['Pooling/Up-sampling 1 time', 'Pooling/Up-sampling 2 times', 'Pooling/Up-sampling 3 times', 'Pooling/Up-sampling 4 times', 'Pooling/Up-sampling 5 times']
fig = plt.figure()
ax = fig.add_subplot(211)
plt.rcParams['axes.unicode_minus'] = False
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_3)
fig.set_size_inches(fig_width, fig_height3)
plt.plot(model_axis, eval_scores_accAllClasses, color=color_depth1, linestyle=linestyle_depth1, linewidth=linewidth_depth1, marker=marker_depth1, markersize=markersize_depth1)
plt.plot(model_axis, eval_scores_accClassesAvg, color=color_depth2, linestyle=linestyle_depth2, linewidth=linewidth_depth2, marker=marker_depth2, markersize=markersize_depth2)
plt.plot(model_axis, eval_scores_mIoU, color=color_depth3, linestyle=linestyle_depth3, linewidth=linewidth_depth3, marker=marker_depth3, markersize=markersize_depth3)
plt.plot(model_axis, eval_scores_wIoU, color=color_depth4, linestyle=linestyle_depth4, linewidth=linewidth_depth4, marker=marker_depth4, markersize=markersize_depth4)
ax.xaxis.set_ticks(itick)
ax.xaxis.set_ticklabels(iticklabel, fontsize=font_normal['size'], color='black', family=font_normal['family'])
for label in ax.xaxis.get_ticklabels():
    label.set_ha('right')
    label.set_rotation(75)
    label.set_rotation_mode('default')  # {None, 'default', 'anchor'} 可注释掉
    label.set_fontsize(font_normal['size'])
    label.set_y(0.03)  # 上下移动
plt.legend(('Accuracy over all classes', 'Average accuracy', 'Average IoU', 'Weighted IoU'), loc='best', fontsize=font_normal['size'], frameon=False, prop=font_normal)
plt.ylabel('Accuracy', font=font_normal, labelpad=labelpad_acc)
plt.xticks(ticks_model, font=font_normal)
plt.yticks(np.linspace(0, 1, num=int((1 - 0) / 0.02 + 1), endpoint=True), font=font_normal)
plt.xlim(xlim_model)
plt.ylim(ylim_accuracy_model)
fname = paper_figure_dir + '/' + 'FigEvalScoresDepth.tif'
if figure_save:
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

plt.show()
