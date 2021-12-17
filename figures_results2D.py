import matplotlib as mpl
import segyio
import matplotlib.colors as mcolors
from pyseismic_local.plot_functions import *
from pyseismic_local.data_read_write_functions import *
from pyseismic_local.evaluation_metrics import *
from pyseismic_local.config import *
################################################################################
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
################################################################################
paper_figure_dir = './' + 'paper_figure'
if not os.path.exists(paper_figure_dir):
    os.mkdir(paper_figure_dir)

dt = 0.0030
nt = 1006
xlines = 1116
ilines = 841

X_test_ID = raw_data_dir + '/' + 'Training_Test1_Test2_Image_1006x1116x841.bin'
X_test = bin_data_reader3D_float32(X_test_ID, nt, xlines, ilines, 1, 1)
X_test = np.reshape(X_test, (nt, xlines, ilines))

Y_pred_ID = module_dir + '/' + 'results/' + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1_Y_pred_1006x1116x841_model3.bin'
Y_pred = bin_data_reader3D_float32(Y_pred_ID, nt, xlines, ilines, 1, 1)
Y_pred = np.reshape(Y_pred, (nt, xlines, ilines))

Y_true = segyio.tools.cube(raw_data_dir + '/' + 'TrainingData_Labels.segy')
Y_true = Y_true.transpose((2, 1, 0)).astype('float32')
print('Y_true.shape:' + str(Y_true.shape))

X_max = 5522.086
X_min = -5195.5234
n_class = 6

zoom_X = 0.1
normY = mpl.colors.Normalize(vmin=1, vmax=6)
normX = mpl.colors.Normalize(vmin=X_min * zoom_X, vmax=X_max * zoom_X)

xlines_true = Y_true.shape[1]
ilines_true = Y_true.shape[2]

time_axis = np.zeros(shape=(nt, 1), dtype='float32')
xline_axis = np.zeros(shape=(xlines, 1), dtype='float32')
iline_axis = np.zeros(shape=(ilines, 1), dtype='float32')
xline_axis_true = np.zeros(shape=(xlines_true, 1), dtype='float32')
iline_axis_true = np.zeros(shape=(ilines_true, 1), dtype='float32')
time_axis[:, 0] = np.array(range(nt)) * dt
xline_axis[:, 0] = np.array(range(xlines)) + 1
iline_axis[:, 0] = np.array(range(ilines)) + 1
xline_axis_true[:, 0] = np.array(range(xlines_true)) + 1
iline_axis_true[:, 0] = np.array(range(ilines_true)) + 1

eval_scores_iline = np.zeros(shape=(ilines_true, 5), dtype=np.float32)
for i_iline in range(ilines_true):
    eval_scores_iline[i_iline, 0] = snr(Y_true[:, 1:, i_iline], Y_pred[:, 1:xlines_true, i_iline])
    accuracy_over_all_classes, accuracy_for_a_class, average_class_accuracy, IoU, average_IoU, weighted_IoU = get_classification_scores(
        Y_true[:, 1:, i_iline], Y_pred[:, 1:xlines_true, i_iline], n_class)
    eval_scores_iline[i_iline, 1] = accuracy_over_all_classes
    eval_scores_iline[i_iline, 2] = average_class_accuracy
    eval_scores_iline[i_iline, 3] = average_IoU
    eval_scores_iline[i_iline, 4] = weighted_IoU

eval_scores_xline = np.zeros(shape=(xlines_true - 1, 5), dtype=np.float32)
for i_xline in range(1, xlines_true):
    eval_scores_xline[i_xline - 1, 0] = snr(Y_true[:, i_xline, :], Y_pred[:, i_xline, 0:ilines_true])
    accuracy_over_all_classes, accuracy_for_a_class, average_class_accuracy, IoU, average_IoU, weighted_IoU = get_classification_scores(
        Y_true[:, i_xline, :], Y_pred[:, i_xline, 0:ilines_true], n_class)
    eval_scores_xline[i_xline - 1, 1] = accuracy_over_all_classes
    eval_scores_xline[i_xline - 1, 2] = average_class_accuracy
    eval_scores_xline[i_xline - 1, 3] = average_IoU
    eval_scores_xline[i_xline - 1, 4] = weighted_IoU

eval_scores_time = np.zeros(shape=(nt, 5), dtype=np.float32)
# for i_time in range(nt):
#     eval_scores_time[i_time, 0] = snr(Y_true[i_time, 1:, :], Y_pred[i_time, 1:xlines_true, 0:ilines_true])
#     accuracy_over_all_classes, accuracy_for_a_class, average_class_accuracy, IoU, average_IoU, weighted_IoU = get_classification_scores(
#         Y_true[i_time, 1:, :], Y_pred[i_time, 1:xlines_true, 0:ilines_true], n_class)
#     eval_scores_time[i_time, 1] = accuracy_over_all_classes
#     eval_scores_time[i_time, 2] = average_class_accuracy
#     eval_scores_time[i_time, 3] = average_IoU
#     eval_scores_time[i_time, 4] = weighted_IoU

# fig = plt.figure()
# plt.plot(eval_scores_iline[:, 3])
#
# fig = plt.figure()
# plt.plot(eval_scores_xline[:, 3])
#
# fig = plt.figure()
# plt.plot(eval_scores_time[:, 3])
#
# print('eval_scores_iline max:' + str(np.max(eval_scores_iline[:, 3])))
# print('eval_scores_xline max:' + str(np.max(eval_scores_xline[:, 3])))

extent_iline = np.min(xline_axis), np.max(xline_axis), np.max(time_axis), np.min(time_axis)
extent_xline = np.min(iline_axis), np.max(iline_axis), np.max(time_axis), np.min(time_axis)
extent_iline_true = np.min(xline_axis_true), np.max(xline_axis_true), np.max(time_axis), np.min(time_axis)
extent_xline_true = np.min(iline_axis_true), np.max(iline_axis_true), np.max(time_axis), np.min(time_axis)
extent_time = np.max(iline_axis), np.min(iline_axis), np.max(xline_axis), np.min(xline_axis)
extent_time_true = np.max(iline_axis_true), np.min(iline_axis_true), np.max(xline_axis_true), np.min(xline_axis_true)
print(extent_iline)
print(extent_xline)
alpha_X = 1.0
alpha_Y = 0.5
labelpad_time_iline = -1
labelpad_time_xline = -1
labelpad_ixline = 0.7
colorbar_pad = 0.0072
font_normal = {'family': 'Arial', 'weight': 'normal', 'size': 8}
xline_ticks = np.hstack((1, (np.linspace(100, 700, num=int((700 - 100) / 100 + 1), endpoint=True)).astype(int),
                         xlines_true,
                         (np.linspace(900, 1000, num=int((1000 - 900) / 100 + 1), endpoint=True)).astype(int), xlines))
iline_ticks = np.hstack(
    (1, (np.linspace(100, 500, num=int((500 - 100) / 100 + 1), endpoint=True)).astype(int), ilines_true, 700, ilines))
print(xline_ticks)
print(iline_ticks)
time_ticks = np.linspace(0, 5, num=int((5 - 0) / 0.2 + 1), endpoint=True)
cmapX = 'gray'
cmapY = mpl.colors.ListedColormap(['red', 'yellow', 'green', 'blue', 'fuchsia', 'brown'])
R = ["Basement/other", "Slope mudstone A", "Mass transport deposit", "Slope mudstone B", "Slope valley",
     "Submarine canyon system"]
font_normal_R = {'family': 'Arial', 'weight': 'normal', 'size': 7}
dpi = 600
axis_linewith = 0.3
figure_save = 1
fig_width = 3.33 * 1.09
fig_height = 4.3
position_fig_h = 100
position_fig_v = 5
d_fig_h = 400
d_fig_v = 500

text_x_iline = 10
text_y_iline = 0.1

text_x_xline = 10
text_y_xline = 0.1

text_x_time = 770
text_y_time = 1056

wm_geometry_1 = '+' + str(position_fig_h + 0 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_2 = '+' + str(position_fig_h + 1 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_3 = '+' + str(position_fig_h + 0 * d_fig_h) + '+' + str(position_fig_v + 1 * d_fig_v)
wm_geometry_4 = '+' + str(position_fig_h + 1 * d_fig_h) + '+' + str(position_fig_v + 1 * d_fig_v)

wm_geometry_5 = '+' + str(position_fig_h + 2 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_6 = '+' + str(position_fig_h + 3 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)

iline_ID = 5 - 1
xline_ID = 3 - 1
time_ID = 646 - 1

linewidth_holdon = 1.0
linestyle_holdon = ':'
linecolor1_holdon = 'black'
linecolor2_holdon = 'white'

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_1)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(X_test[:, :, iline_ID].reshape(nt, xlines), cmap=cmapX, norm=normX, aspect='auto', alpha=alpha_X,
           origin='upper', extent=extent_iline)
plt.imshow(Y_true[:, 0:xlines_true, iline_ID].reshape(nt, xlines_true), cmap=cmapY, norm=normY, aspect='auto',
           alpha=alpha_Y, origin='upper', extent=extent_iline_true)
# plt.plot((xline_ID + 1) * np.ones(shape=(nt, 1), dtype=np.float32), time_axis, color=linecolor1_holdon, linestyle=linestyle_holdon, linewidth=linewidth_holdon)
plt.text(400, 2.80, R[0], rotation=0, fontdict=font_normal_R, color='white')
plt.text(450, 2.45, R[1], rotation=0, fontdict=font_normal_R, color='black')
plt.text(400, 1.00, R[1], rotation=0, fontdict=font_normal_R, color='black')
plt.text(400, 2.27, R[2], rotation=345, fontdict=font_normal_R, color='white')
plt.text(400, 1.52, R[3], rotation=0, fontdict=font_normal_R, color='white')
plt.text(400, 0.20, R[3], rotation=0, fontdict=font_normal_R, color='white')
plt.text(200, 1.25, R[4], rotation=355, fontdict=font_normal_R, color='black')
plt.text(300, 0.62, R[5], rotation=0, fontdict=font_normal_R, color='white')
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Crossline number', font=font_normal, labelpad=labelpad_ixline)
plt.ylabel('Time (s)', font=font_normal, labelpad=labelpad_time_iline)
plt.xticks(xline_ticks, font=font_normal)
plt.yticks(time_ticks, font=font_normal)
plt.xlim(np.min(xline_axis) - 1, np.max(xline_axis) + 1)
plt.ylim(np.max(time_axis) + 1 * dt, np.min(time_axis) - 1 * dt)
colorbar_Y(ncolors=6, cmap=cmapY, orientation='horizontal', tick=np.array(range(6)) + 1, font=font_normal,
           pad=colorbar_pad)
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)
plt.tight_layout()

if figure_save:
    fname = paper_figure_dir + '/' + 'FigDemoIlineSEG.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')
#
fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_2)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(X_test[:, :, iline_ID].reshape(nt, xlines), cmap=cmapX, norm=normX, aspect='auto', alpha=alpha_X,
           origin='upper', extent=extent_iline)
plt.imshow(Y_pred[:, :, iline_ID].reshape(nt, xlines), cmap=cmapY, norm=normY, aspect='auto', alpha=alpha_Y,
           origin='upper', extent=extent_iline)
# plt.plot((xline_ID + 1) * np.ones(shape=(nt, 1), dtype=np.float32), time_axis, color=linecolor1_holdon, linestyle=linestyle_holdon, linewidth=linewidth_holdon)
plt.text(400, 2.80, R[0], rotation=0, fontdict=font_normal_R, color='white')
plt.text(450, 2.45, R[1], rotation=0, fontdict=font_normal_R, color='black')
plt.text(400, 1.00, R[1], rotation=0, fontdict=font_normal_R, color='black')
plt.text(400, 2.27, R[2], rotation=345, fontdict=font_normal_R, color='white')
plt.text(400, 1.52, R[3], rotation=0, fontdict=font_normal_R, color='white')
plt.text(400, 0.20, R[3], rotation=0, fontdict=font_normal_R, color='white')
plt.text(200, 1.25, R[4], rotation=355, fontdict=font_normal_R, color='black')
plt.text(400, 0.65, R[5], rotation=0, fontdict=font_normal_R, color='white')
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Crossline number', font=font_normal, labelpad=labelpad_ixline)
plt.ylabel('Time (s)', font=font_normal, labelpad=labelpad_time_iline)
plt.xticks(xline_ticks, font=font_normal)
plt.yticks(time_ticks, font=font_normal)
plt.text(text_x_iline, text_y_iline,
         'Inline ' + str(iline_ID + 1) + ', mIoU = %2.4f' % (eval_scores_iline[iline_ID, 3]), fontdict=font_normal,
         color='yellow')
plt.xlim(np.min(xline_axis) - 1, np.max(xline_axis) + 1)
plt.ylim(np.max(time_axis) + 1 * dt, np.min(time_axis) - 1 * dt)
colorbar_Y(ncolors=6, cmap=cmapY, orientation='horizontal', tick=np.array(range(6)) + 1, font=font_normal,
           pad=colorbar_pad)
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)
plt.tight_layout()
if figure_save:
    fname = paper_figure_dir + '/' + 'FigDemoIlineWeDo.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_3)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(X_test[:, xline_ID, :].reshape(nt, ilines), cmap=cmapX, norm=normX, aspect='auto', alpha=alpha_X,
           origin='upper', extent=extent_xline)
plt.imshow(Y_true[:, xline_ID, :].reshape(nt, ilines_true), cmap=cmapY, norm=normY, aspect='auto', alpha=alpha_Y,
           origin='upper', extent=extent_xline_true)
plt.text(350, 2.80, R[0], rotation=0, fontdict=font_normal_R, color='white')
plt.text(350, 2.1, R[1], rotation=0, fontdict=font_normal_R, color='black')
plt.text(350, 0.8, R[1], rotation=0, fontdict=font_normal_R, color='black')
plt.text(300, 1.9, R[2], rotation=0, fontdict=font_normal_R, color='white')
plt.text(350, 1.2, R[3], rotation=0, fontdict=font_normal_R, color='white')
plt.text(350, 0.20, R[3], rotation=0, fontdict=font_normal_R, color='white')
plt.text(0, 0.75, R[5], rotation=345, fontdict=font_normal_R, color='white',fontsize=6)
# plt.plot((iline_ID + 1) * np.ones(shape=(nt, 1), dtype=np.float32), time_axis, color=linecolor2_holdon, linestyle=linestyle_holdon, linewidth=linewidth_holdon)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Inline number', font=font_normal, labelpad=labelpad_ixline)
plt.ylabel('Time (s)', font=font_normal, labelpad=labelpad_time_xline)
plt.xticks(iline_ticks, font=font_normal)
plt.yticks(time_ticks, font=font_normal)
plt.xlim(np.min(iline_axis) - 1, np.max(iline_axis) + 1)
plt.ylim(np.max(time_axis) + 1 * dt, np.min(time_axis) - 1 * dt)
colorbar_Y(ncolors=6, cmap=cmapY, orientation='horizontal', tick=np.array(range(6)) + 1, font=font_normal,
           pad=colorbar_pad)
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)
plt.tight_layout()
if figure_save:
    fname = paper_figure_dir + '/' + 'FigDemoXlineSEG.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_4)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(X_test[:, xline_ID, :].reshape(nt, ilines), cmap=cmapX, norm=normX, aspect='auto', alpha=alpha_X,
           origin='upper', extent=extent_xline)
plt.imshow(Y_pred[:, xline_ID, :].reshape(nt, ilines), cmap=cmapY, norm=normY, aspect='auto', alpha=alpha_Y,
           origin='upper', extent=extent_xline)
# plt.plot((iline_ID + 1) * np.ones(shape=(nt, 1), dtype=np.float32), time_axis, color=linecolor2_holdon, linestyle=linestyle_holdon, linewidth=linewidth_holdon)
plt.text(350, 2.80, R[0], rotation=0, fontdict=font_normal_R, color='white')
plt.text(350, 2.1, R[1], rotation=0, fontdict=font_normal_R, color='black')
plt.text(350, 0.8, R[1], rotation=0, fontdict=font_normal_R, color='black')
plt.text(300, 1.9, R[2], rotation=0, fontdict=font_normal_R, color='white')
plt.text(350, 1.2, R[3], rotation=0, fontdict=font_normal_R, color='white')
plt.text(350, 0.20, R[3], rotation=0, fontdict=font_normal_R, color='white')
plt.text(480, 0.53, R[5], rotation=0, fontdict=font_normal_R, color='white')
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Inline number', font=font_normal, labelpad=labelpad_ixline)
plt.ylabel('Time (s)', font=font_normal, labelpad=labelpad_time_xline)
plt.xticks(iline_ticks, font=font_normal)
plt.yticks(time_ticks, font=font_normal)
plt.text(text_x_xline, text_y_xline,
         'Crossline ' + str(xline_ID + 1) + ', mIoU = %2.4f' % (eval_scores_xline[xline_ID - 1, 3]),
         fontdict=font_normal, color='yellow')
plt.xlim(np.min(iline_axis) - 1, np.max(iline_axis) + 1)
plt.ylim(np.max(time_axis) + 1 * dt, np.min(time_axis) - 1 * dt)
colorbar_Y(ncolors=6, cmap=cmapY, orientation='horizontal', tick=np.array(range(6)) + 1, font=font_normal,
           pad=colorbar_pad)
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)
plt.tight_layout()
if figure_save:
    fname = paper_figure_dir + '/' + 'FigDemoXlineWeDo.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_5)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(np.fliplr(X_test[time_ID, :, :].reshape(xlines, ilines)), cmap=cmapX, norm=normX, aspect='auto', alpha=alpha_X, origin='upper', extent=extent_time)
plt.imshow(np.fliplr(Y_true[time_ID, :, :].reshape(xlines_true, ilines_true)), cmap=cmapY, norm=normY, aspect='auto', alpha=alpha_Y, origin='upper', extent=extent_time_true)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Inline number', font=font_normal, labelpad=labelpad_ixline)
plt.ylabel('Crossline number', font=font_normal, labelpad=labelpad_time_xline)
plt.xticks(iline_ticks, font=font_normal)
plt.yticks(xline_ticks, font=font_normal)
plt.xlim(np.max(iline_axis) + 1, np.min(iline_axis) - 1)
plt.ylim(np.max(xline_axis) + 1, np.min(xline_axis) - 1)
colorbar_Y(ncolors=6, cmap=cmapY, orientation='horizontal', tick=np.array(range(6)) + 1, font=font_normal, pad=colorbar_pad)
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)
plt.tight_layout()
if figure_save:
    fname = paper_figure_dir + '/' + 'FigDemoTimeSEG.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_6)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(np.fliplr(X_test[time_ID, :, :].reshape(xlines, ilines)), cmap=cmapX, norm=normX, aspect='auto', alpha=alpha_X, origin='upper', extent=extent_time)
plt.imshow(np.fliplr(Y_pred[time_ID, :, :].reshape(xlines, ilines)), cmap=cmapY, norm=normY, aspect='auto', alpha=alpha_Y, origin='upper', extent=extent_time)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Inline number', font=font_normal, labelpad=labelpad_ixline)
plt.ylabel('Crossline number', font=font_normal, labelpad=labelpad_time_xline)
plt.xticks(iline_ticks, font=font_normal)
plt.yticks(xline_ticks, font=font_normal)
plt.text(text_x_time, text_y_time, 'Time slice at $t$ = ' + str(time_ID * dt) + ' s, mIoU = %2.4f' % (eval_scores_time[time_ID, 3]), fontdict=font_normal, color='yellow')
plt.xlim(np.max(iline_axis) + 1, np.min(iline_axis) - 1)
plt.ylim(np.max(xline_axis) + 1, np.min(xline_axis) - 1)
colorbar_Y(ncolors=6, cmap=cmapY, orientation='horizontal', tick=np.array(range(6)) + 1, font=font_normal, pad=colorbar_pad)
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)
plt.tight_layout()
if figure_save:
    fname = paper_figure_dir + '/' + 'FigDemoTimeWeDo.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

plt.show()
