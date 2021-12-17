import sys

sys.path.append('./iPython')
import matplotlib as mpl
import segyio
import matplotlib.colors as mcolors
from iPython.plot_functions import *
from iPython.data_read_write_functions import *
################################################################################
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
################################################################################
paper_figure_dir = './' + 'paper_figure'
if not os.path.exists(paper_figure_dir):
    os.mkdir(paper_figure_dir)

raw_data_X_ID = []
raw_data_Y_ID = []

raw_data_X_ID.append('D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' + 'TrainingData_Image.segy')
raw_data_Y_ID.append('D:/BigData/SeismicDataField/SEG2020MLInterpretationWorkshop/' + 'TrainingData_Labels.segy')

Y_pred_ID = 'D:/BigData/AI_SeismicFaciesClassification/results/' + '3_All_iline_patchR992C576_batch16_number10000_stride45_5_loss5_lr100_pool1_kernelSize11_Net5_kernels16_repeat1_dila1_repro2_Y_pred_1006x1116x841_model3.bin'
Y_pred = bin_data_reader3D_float32(Y_pred_ID, 1006, 1116, 841, 1, 1)
Y_pred = np.reshape(Y_pred, (1006, 1116, 841))

Y_pred = Y_pred[:, 0, 0:590]

X = segyio.tools.cube(raw_data_X_ID[0])
Y = segyio.tools.cube(raw_data_Y_ID[0])
X = X.transpose((2, 1, 0)).astype('float32')
Y = Y.transpose((2, 1, 0)).astype('float32')

X_max = 5522.086
X_min = -5195.5234

figure_save = 1
zoom_X = 0.1
normY = mpl.colors.Normalize(vmin=1, vmax=6)
normX = mpl.colors.Normalize(vmin=X_min * zoom_X, vmax=X_max * zoom_X)

dt = 0.0030
nt = X.shape[0]
ilines = X.shape[2]
time_axis = np.zeros(shape=(nt, 1), dtype='float32')
iline_axis = np.zeros(shape=(ilines, 1), dtype='float32')
time_axis[:, 0] = np.array(range(nt)) * dt
iline_axis[:, 0] = np.array(range(ilines)) + 1

extent_xline = np.min(iline_axis), np.max(iline_axis), np.max(time_axis), np.min(time_axis)
print(extent_xline)
alpha_X = 1.0
alpha_Y = 1.0
labelpad_time = -1
labelpad_iline = 0.7
colorbar_pad = 0.0072
font_normal = {'family': 'Arial', 'weight': 'normal', 'size': 8}
iline_ticks = np.hstack((1, (np.linspace(100, 500, num=int((500 - 100) / 100 + 1), endpoint=True)).astype(int), ilines))
print(iline_ticks)
time_ticks = np.linspace(0, 5, num=int((5 - 0) / 0.2 + 1), endpoint=True)
cmapX = we_seismic_cmap()
cmapY = mpl.colors.ListedColormap(['red', 'yellow', 'green', 'blue', 'fuchsia', 'brown'])

axis_linewith = 0.3
fig_width = 3.33 * 0.59
fig_height = 3.999
position_fig_h = 100
position_fig_v = 50
d_fig_h = 230
d_fig_v = 400
dpi = 300
wm_geometry_1 = '+' + str(position_fig_h + 0 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_2 = '+' + str(position_fig_h + 1 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_3 = '+' + str(position_fig_h + 2 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_4 = '+' + str(position_fig_h + 3 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_5 = '+' + str(position_fig_h + 0 * d_fig_h) + '+' + str(position_fig_v + 1 * d_fig_v)
wm_geometry_6 = '+' + str(position_fig_h + 1 * d_fig_h) + '+' + str(position_fig_v + 1 * d_fig_v)
wm_geometry_7 = '+' + str(position_fig_h + 2 * d_fig_h) + '+' + str(position_fig_v + 1 * d_fig_v)
wm_geometry_8 = '+' + str(position_fig_h + 3 * d_fig_h) + '+' + str(position_fig_v + 1 * d_fig_v)

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_1)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(X[:, 0, :].reshape(nt, ilines), cmap=cmapX, norm=normX, aspect='auto', alpha=alpha_X, origin='upper', extent=extent_xline)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Inline number', font=font_normal, labelpad=labelpad_iline)
plt.ylabel('Time (s)', font=font_normal, labelpad=labelpad_time)
plt.xticks(iline_ticks, font=font_normal)
plt.yticks(time_ticks, font=font_normal)
plt.xlim(np.min(iline_axis) - 1, np.max(iline_axis) + 1)
plt.ylim(np.max(time_axis) + 1 * dt, np.min(time_axis) - 1 * dt)
colorbar_X(orientation='horizontal', font=font_normal, pad=colorbar_pad)
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)
plt.tight_layout()
if figure_save:
    fname = paper_figure_dir + '/' + 'FigXline01X.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_2)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(X[:, 1, :].reshape(nt, ilines), cmap=cmapX, norm=normX, aspect='auto', alpha=alpha_X, origin='upper', extent=extent_xline)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Inline number', font=font_normal, labelpad=labelpad_iline)
plt.ylabel('Time (s)', font=font_normal, labelpad=labelpad_time)
plt.xticks(iline_ticks, font=font_normal)
plt.yticks(time_ticks, font=font_normal)
plt.xlim(np.min(iline_axis) - 1, np.max(iline_axis) + 1)
plt.ylim(np.max(time_axis) + 1 * dt, np.min(time_axis) - 1 * dt)
colorbar_X(orientation='horizontal', font=font_normal, pad=colorbar_pad)
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)
plt.tight_layout()
if figure_save:
    fname = paper_figure_dir + '/' + 'FigXline02X.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_3)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(X[:, 2, :].reshape(nt, ilines), cmap=cmapX, norm=normX, aspect='auto', alpha=alpha_X, origin='upper', extent=extent_xline)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Inline number', font=font_normal, labelpad=labelpad_iline)
plt.ylabel('Time (s)', font=font_normal, labelpad=labelpad_time)
plt.xticks(iline_ticks, font=font_normal)
plt.yticks(time_ticks, font=font_normal)
plt.xlim(np.min(iline_axis) - 1, np.max(iline_axis) + 1)
plt.ylim(np.max(time_axis) + 1 * dt, np.min(time_axis) - 1 * dt)
colorbar_X(orientation='horizontal', font=font_normal, pad=colorbar_pad)
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)
plt.tight_layout()
if figure_save:
    fname = paper_figure_dir + '/' + 'FigXline03X.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_4)
fig.set_size_inches(fig_width, fig_height)
plt.imshow((X[:, 1, :] - X[:, 0, :]).reshape(nt, ilines), cmap=cmapX, norm=normX, aspect='auto', alpha=alpha_X, origin='upper', extent=extent_xline)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Inline number', font=font_normal, labelpad=labelpad_iline)
plt.ylabel('Time (s)', font=font_normal, labelpad=labelpad_time)
plt.xticks(iline_ticks, font=font_normal)
plt.yticks(time_ticks, font=font_normal)
plt.xlim(np.min(iline_axis) - 1, np.max(iline_axis) + 1)
plt.ylim(np.max(time_axis) + 1 * dt, np.min(time_axis) - 1 * dt)
colorbar_X(orientation='horizontal', font=font_normal, pad=colorbar_pad)
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)
plt.tight_layout()
if figure_save:
    fname = paper_figure_dir + '/' + 'FigXline0102XDiff.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_5)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(Y[:, 0, :].reshape(nt, ilines), cmap=cmapY, norm=normY, aspect='auto', alpha=alpha_Y, origin='upper', extent=extent_xline)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Inline number', font=font_normal, labelpad=labelpad_iline)
plt.ylabel('Time (s)', font=font_normal, labelpad=labelpad_time)
plt.xticks(iline_ticks, font=font_normal)
plt.yticks(time_ticks, font=font_normal)
plt.xlim(np.min(iline_axis) - 1, np.max(iline_axis) + 1)
plt.ylim(np.max(time_axis) + 1 * dt, np.min(time_axis) - 1 * dt)
colorbar_Y(ncolors=6, cmap=cmapY, orientation='horizontal', tick=np.array(range(6)) + 1, font=font_normal, pad=colorbar_pad)
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)
plt.tight_layout()
if figure_save:
    fname = paper_figure_dir + '/' + 'FigXline01Y.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_6)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(Y[:, 1, :].reshape(nt, ilines), cmap=cmapY, norm=normY, aspect='auto', alpha=alpha_Y, origin='upper', extent=extent_xline)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Inline number', font=font_normal, labelpad=labelpad_iline)
plt.ylabel('Time (s)', font=font_normal, labelpad=labelpad_time)
plt.xticks(iline_ticks, font=font_normal)
plt.yticks(time_ticks, font=font_normal)
plt.xlim(np.min(iline_axis) - 1, np.max(iline_axis) + 1)
plt.ylim(np.max(time_axis) + 1 * dt, np.min(time_axis) - 1 * dt)
colorbar_Y(ncolors=6, cmap=cmapY, orientation='horizontal', tick=np.array(range(6)) + 1, font=font_normal, pad=colorbar_pad)
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)
plt.tight_layout()
if figure_save:
    fname = paper_figure_dir + '/' + 'FigXline02Y.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_7)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(Y[:, 2, :].reshape(nt, ilines), cmap=cmapY, norm=normY, aspect='auto', alpha=alpha_Y, origin='upper', extent=extent_xline)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Inline number', font=font_normal, labelpad=labelpad_iline)
plt.ylabel('Time (s)', font=font_normal, labelpad=labelpad_time)
plt.xticks(iline_ticks, font=font_normal)
plt.yticks(time_ticks, font=font_normal)
plt.xlim(np.min(iline_axis) - 1, np.max(iline_axis) + 1)
plt.ylim(np.max(time_axis) + 1 * dt, np.min(time_axis) - 1 * dt)
colorbar_Y(ncolors=6, cmap=cmapY, orientation='horizontal', tick=np.array(range(6)) + 1, font=font_normal, pad=colorbar_pad)
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)
plt.tight_layout()
if figure_save:
    fname = paper_figure_dir + '/' + 'FigXline03Y.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_8)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(Y_pred.reshape(nt, ilines), cmap=cmapY, norm=normY, aspect='auto', alpha=alpha_Y, origin='upper', extent=extent_xline)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Inline number', font=font_normal, labelpad=labelpad_iline)
plt.ylabel('Time (s)', font=font_normal, labelpad=labelpad_time)
plt.xticks(iline_ticks, font=font_normal)
plt.yticks(time_ticks, font=font_normal)
plt.xlim(np.min(iline_axis) - 1, np.max(iline_axis) + 1)
plt.ylim(np.max(time_axis) + 1 * dt, np.min(time_axis) - 1 * dt)
colorbar_Y(ncolors=6, cmap=cmapY, orientation='horizontal', tick=np.array(range(6)) + 1, font=font_normal, pad=colorbar_pad)
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)
plt.tight_layout()
if figure_save:
    fname = paper_figure_dir + '/' + 'FigXline01Ypred.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

plt.show()
