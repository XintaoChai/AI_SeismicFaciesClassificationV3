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

log_pool1 = pd.read_csv(log_root_dir + 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1_dila1_repro2' + '/log.csv', header=0)
log_pool2 = pd.read_csv(log_root_dir + 'patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool2_kernelSize11_Net5_kernels16_repeat1' + '/log.csv', header=0)

paper_results_dir = 'D:/Nutstore/AI_ML_DL_WeDo/AI_SeismicFaciesClassification/paper_results/'

eval_scores_pool1_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1_dila1_repro2_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_pool1 = bin_data_reader3D_float32(eval_scores_pool1_ID, 1, 5, 1, 1, 1)
eval_scores_pool1 = np.reshape(eval_scores_pool1, (1, 5, 1))
print('eval_scores_pool1: ' + str(eval_scores_pool1))

eval_scores_pool2_ID = paper_results_dir + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool2_kernelSize11_Net5_kernels16_repeat1_models3_3_cube_eval_scores_1x5x1.bin'
eval_scores_pool2 = bin_data_reader3D_float32(eval_scores_pool2_ID, 1, 5, 1, 1, 1)
eval_scores_pool2 = np.reshape(eval_scores_pool2, (1, 5, 1))
print('eval_scores_pool2: ' + str(eval_scores_pool2))

figure_save = 1
dpi = 300

ylim_accuracy_pool = (0.893999, 1.003999)
ticks_epoch = [1, 2, 3]
ticks_model = [1, 2, 3, 4, 5]
xlim_epoch = (1 - 0.05, 3 + 0.05)
xlim_pool = (-1, 2)
epoch_axis = np.zeros(shape=(epoch_max, 1), dtype='int32')
epoch_axis[:, 0] = np.array(range(epoch_max)) + 1
model_axis = np.zeros(shape=(2, 1), dtype='int32')
model_axis[:, 0] = np.array(range(2)) + 1

eval_scores_accAllClasses = np.zeros(shape=(2, 1), dtype='float32')
accAllClasses_index = 1
eval_scores_accAllClasses[0, 0] = eval_scores_pool1[0, accAllClasses_index, 0]
eval_scores_accAllClasses[1, 0] = eval_scores_pool2[0, accAllClasses_index, 0]

eval_scores_accClassesAvg = np.zeros(shape=(2, 1), dtype='float32')
accClassesAvg_index = 2
eval_scores_accClassesAvg[0, 0] = eval_scores_pool1[0, accClassesAvg_index, 0]
eval_scores_accClassesAvg[1, 0] = eval_scores_pool2[0, accClassesAvg_index, 0]

eval_scores_mIoU = np.zeros(shape=(2, 1), dtype='float32')
mIoU_index = 3
eval_scores_mIoU[0, 0] = eval_scores_pool1[0, mIoU_index, 0]
eval_scores_mIoU[1, 0] = eval_scores_pool2[0, mIoU_index, 0]

eval_scores_wIoU = np.zeros(shape=(2, 1), dtype='float32')
wIoU_index = 4
eval_scores_wIoU[0, 0] = eval_scores_pool1[0, wIoU_index, 0]
eval_scores_wIoU[1, 0] = eval_scores_pool2[0, wIoU_index, 0]

color_pool2 = 'black'
marker_pool2 = 's'
linestyle_pool2 = '-'
linewidth_pool2 = 0.5
markersize_pool2 = 5

color_loss2 = 'magenta'
marker_loss2 = 'o'
linestyle_loss2 = '-'
linewidth_loss2 = linewidth_pool2
markersize_loss2 = markersize_pool2

color_loss3 = 'blue'
marker_loss3 = '*'
linestyle_loss3 = '-'
linewidth_loss3 = linewidth_pool2
markersize_loss3 = markersize_pool2

color_loss4 = 'green'
marker_loss4 = 'p'
linestyle_loss4 = '-'
linewidth_loss4 = linewidth_pool2
markersize_loss4 = markersize_pool2

color_pool1 = 'red'
marker_pool1 = 'x'
linestyle_pool1 = '-'
linewidth_pool1 = linewidth_pool2
markersize_pool1 = markersize_pool2

labelpad_epoch_loss = -1
labelpad_epoch_acc = labelpad_epoch_loss
labelpad_loss = 0
labelpad_acc = labelpad_loss
font_normal = {'family': 'Arial', 'weight': 'normal', 'size': 8}

axis_linewith = 0.3
fig_width = 3.33 * 1.12311
fig_height = 7.5
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
# fig.set_size_inches(fig_width, fig_height)
# plt.plot(epoch_axis, np.log10(log_pool1['loss'].values[0:epoch_max] / log_pool1['loss'].values[0]), color=color_pool1, linestyle=linestyle_pool1, linewidth=linewidth_pool1, marker=marker_pool1,
#          markersize=markersize_pool1)
# plt.plot(epoch_axis, np.log10(log_pool2['loss'].values[0:epoch_max] / log_pool2['loss'].values[0]), color=color_pool2, linestyle=linestyle_pool2, linewidth=linewidth_pool2, marker=marker_pool2,
#          markersize=markersize_pool2)
# plt.legend(('Max pooling', 'Average pooling'), loc='best', fontsize=font_normal['size'], frameon=False)
# plt.xlabel('Epoch', font=font_normal, labelpad=labelpad_epoch_loss)
# plt.ylabel('Loss (normalized)', font=font_normal, labelpad=labelpad_loss)
# plt.xticks(ticks_epoch, font=font_normal)
# plt.yticks(np.linspace(-2.1, 0, num=int((0 - (-2.1)) / 0.1 + 1), endpoint=True), font=font_normal)
# plt.xlim(xlim_epoch)
# plt.ylim(ylim_loss_train)
# fname = paper_figure_dir + '/' + 'FigTrainLossLoss.tif'
# if figure_save:
#     plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')
#
# fig = plt.figure()
# plt.rcParams['axes.unicode_minus'] = False
# mngr = plt.get_current_fig_manager()
# mngr.window.wm_geometry(wm_geometry_2)
# fig.set_size_inches(fig_width, fig_height)
# plt.plot(epoch_axis, log_pool1['accuracy'].values[0:epoch_max], color=color_pool1, linestyle=linestyle_pool1, linewidth=linewidth_pool1, marker=marker_pool1, markersize=markersize_pool1)
# plt.plot(epoch_axis, log_pool2['accuracy'].values[0:epoch_max], color=color_pool2, linestyle=linestyle_pool2, linewidth=linewidth_pool2, marker=marker_pool2, markersize=markersize_pool2)
# plt.legend(('Max pooling', 'Average pooling'), loc='best', fontsize=font_normal['size'], frameon=False)
# plt.xlabel('Epoch', font=font_normal, labelpad=labelpad_epoch_acc)
# plt.ylabel('accuracy', font=font_normal, labelpad=labelpad_acc)
# plt.xticks(ticks_epoch, font=font_normal)
# plt.yticks(np.linspace(0, 1, num=int((1 - 0) / 0.02 + 1), endpoint=True), font=font_normal)
# plt.xlim(xlim_epoch)
# plt.ylim(ylim_accuracy_train)
# fname = paper_figure_dir + '/' + 'FigTrainAccPool.tif'
# if figure_save:
#     plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

itick = [0, 1]
iticklabel = ['Max pooling', 'Average pooling']
fig = plt.figure()
ax = fig.add_subplot(211)
plt.rcParams['axes.unicode_minus'] = False
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_3)
fig.set_size_inches(fig_width, fig_height)
plt.plot(eval_scores_accAllClasses, color=color_pool2, linestyle=linestyle_pool2, linewidth=linewidth_pool2, marker=marker_pool2, markersize=markersize_pool2)
plt.plot(eval_scores_accClassesAvg, color=color_loss2, linestyle=linestyle_loss2, linewidth=linewidth_loss2, marker=marker_loss2, markersize=markersize_loss2)
plt.plot(eval_scores_mIoU, color=color_loss3, linestyle=linestyle_loss3, linewidth=linewidth_loss3, marker=marker_loss3, markersize=markersize_loss3)
plt.plot(eval_scores_wIoU, color=color_loss4, linestyle=linestyle_loss4, linewidth=linewidth_loss4, marker=marker_loss4, markersize=markersize_loss4)
ax.xaxis.set_ticks(itick)
ax.xaxis.set_ticklabels(iticklabel, fontsize=font_normal['size'], color='black', family=font_normal['family'])
for label in ax.xaxis.get_ticklabels():
    label.set_ha('right')
    label.set_rotation(56)
    label.set_rotation_mode('default')  # {None, 'default', 'anchor'} 可注释掉
    label.set_fontsize(font_normal['size'])
    label.set_y(0.03)  # 上下移动
plt.legend(('Accuracy over all classes', 'Average accuracy', 'Average IoU', 'Weighted IoU'), loc='best', fontsize=font_normal['size'], frameon=False, prop=font_normal)
plt.ylabel('Accuracy', font=font_normal, labelpad=labelpad_acc)
plt.xticks(font=font_normal)
plt.yticks(np.linspace(0, 1, num=int((1 - 0) / 0.01 + 1), endpoint=True), font=font_normal)
plt.xlim(xlim_pool)
plt.ylim(ylim_accuracy_pool)
fname = paper_figure_dir + '/' + 'FigEvalScoresPool.tif'
if figure_save:
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

plt.show()
