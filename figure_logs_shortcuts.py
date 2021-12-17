from pyseismic_local.plot_functions import *
from pyseismic_local.data_read_write_functions import *
from pyseismic.image_processing import *
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
epoch_max = 3

log_lr001 = pd.read_csv(
    log_root_dir + 'patchR0992C0576_batch016_number10000_stride8_8_lr1000_kSize11_ResCase0_ResNo1_Net5_Crep2_Repro01_kernels16_32_64_128_256_512' + '/log.csv', header=0)
log_lr010 = pd.read_csv(
    log_root_dir + 'patchR0992C0576_batch016_number10000_stride8_8_lr1000_kSize11_ResCase3_ResNo1_Net5_Crep2_Repro06_kernels16_32_64_128_256_512' + '/log.csv', header=0)

figure_save = 1
dpi = 600

ylim_loss_train = (-1.1, 0.02)
ylim_accuracy_train = (0.8, 1.008)
ticks_epoch = list(range(1, 11))
xlim_epoch = (1 - 0.2, epoch_max + 0.2)
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

labelpad_epoch_loss = -1.5
labelpad_epoch_acc = -1.5
labelpad_loss = 0
labelpad_acc = labelpad_loss
font_normal = {'family': 'Arial', 'weight': 'normal', 'size': 8}

axis_linewith = 0.3
fig_width = 3.33 * 1.12311
fig_height1 = 3
fig_height2 = 2.0
position_fig_h = 100
position_fig_v = 50
d_fig_h = 400
d_fig_v = 400
wm_geometry_1 = '+' + str(position_fig_h + 0 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_2 = '+' + str(position_fig_h + 1 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)

fig = plt.figure()
ax = fig.add_subplot(211)
plt.rcParams['axes.unicode_minus'] = False
mngr = plt.get_current_fig_manager()
# mngr.window.wm_geometry(wm_geometry_1)
fig.set_size_inches(fig_width, fig_height1)
plt.plot(epoch_axis, np.log10(log_lr001['loss'].values[0:epoch_max] / log_lr001['loss'].values[0]), color=color_lr001, linestyle=linestyle_lr001, linewidth=linewidth_lr001,
         marker=marker_lr001, markersize=markersize_lr001)
plt.plot(epoch_axis, np.log10(log_lr010['loss'].values[0:epoch_max] / log_lr010['loss'].values[0]), color=color_lr010, linestyle=linestyle_lr010, linewidth=linewidth_lr010,
         marker=marker_lr010, markersize=markersize_lr010)
plt.legend(('Without shortcut connections', 'With shortcut connections'), loc='upper right', fontsize=font_normal['size'], frameon=False, prop=font_normal)
plt.xlabel('Epoch', font=font_normal, labelpad=labelpad_epoch_loss)
plt.ylabel('log$_{10}$(normalized loss)', font=font_normal, labelpad=labelpad_loss)
plt.xticks(ticks_epoch, font=font_normal)
plt.yticks(np.linspace(-1.5, 0, num=int((0 - (-1.5)) / 0.1 + 1), endpoint=True), font=font_normal)
plt.xlim(xlim_epoch)
plt.ylim(ylim_loss_train)
if figure_save:
    fname = paper_figure_dir + '/' + 'FigTrainLossShortcuts.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')
    figure_cutter(fname, fname, loc=(0, 0))

fig = plt.figure()
ax = fig.add_subplot(211)
plt.rcParams['axes.unicode_minus'] = False
mngr = plt.get_current_fig_manager()
# mngr.window.wm_geometry(wm_geometry_2)
fig.set_size_inches(fig_width, fig_height2)
plt.plot(epoch_axis, log_lr001['accuracy'].values[0:epoch_max], color=color_lr001, linestyle=linestyle_lr001, linewidth=linewidth_lr001, marker=marker_lr001,
         markersize=markersize_lr001)
plt.plot(epoch_axis, log_lr010['accuracy'].values[0:epoch_max], color=color_lr010, linestyle=linestyle_lr010, linewidth=linewidth_lr010, marker=marker_lr010,
         markersize=markersize_lr010)
plt.legend(('Without shortcut connections', 'With shortcut connections'), loc='lower right', fontsize=font_normal['size'], frameon=False, prop=font_normal)
plt.xlabel('Epoch', font=font_normal, labelpad=labelpad_epoch_acc)
plt.ylabel('Accuracy', font=font_normal, labelpad=labelpad_acc)
plt.xticks(ticks_epoch, font=font_normal)
plt.yticks(np.linspace(0, 1, num=int((1 - 0) / 0.05 + 1), endpoint=True), font=font_normal)
plt.xlim(xlim_epoch)
plt.ylim(ylim_accuracy_train)
if figure_save:
    fname = paper_figure_dir + '/' + 'FigTrainAccShortcuts.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')
    figure_cutter(fname, fname, loc=(0, 0))

plt.show()
