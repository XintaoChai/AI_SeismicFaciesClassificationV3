import matplotlib as mpl
import segyio
import matplotlib.colors as mcolors
from pyseismic_local.plot_functions import *
from pyseismic_local.config import *
################################################################################
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
################################################################################
paper_figure_dir = './' + 'paper_figure'
if not os.path.exists(paper_figure_dir):
    os.mkdir(paper_figure_dir)

R = ["Basement/other", "Slope mudstone A", "Mass transport deposit", "Slope mudstone B", "Slope valley",
     "Submarine canyon system"]
font_normal_R = {'family': 'Arial', 'weight': 'normal', 'size': 7}
raw_data_X_ID = []
raw_data_Y_ID = []

raw_data_X_ID.append(raw_data_dir + '/' + 'TrainingData_Image.segy')
raw_data_Y_ID.append(raw_data_dir + '/' + 'TrainingData_Labels.segy')

X = segyio.tools.cube(raw_data_X_ID[0])
Y = segyio.tools.cube(raw_data_Y_ID[0])
X = X.transpose((2, 1, 0)).astype('float32')
Y = Y.transpose((2, 1, 0)).astype('float32')
print('Y.shape:' + str(Y.shape))
patch_size_height = Y.shape[0]
patch_size_width = Y.shape[1]

X_max = 5522.086
X_min = -5195.5234

figure_save = 1
zoom_X = 0.1
normY = mpl.colors.Normalize(vmin=1, vmax=6)
normX = mpl.colors.Normalize(vmin=X_min * zoom_X, vmax=X_max * zoom_X)

dt = 0.0030
nt = X.shape[0]
xlines = X.shape[1]
ilines = X.shape[2]
time_axis = np.zeros(shape=(nt, 1), dtype='float32')
xline_axis = np.zeros(shape=(xlines, 1), dtype='float32')
iline_axis = np.zeros(shape=(ilines, 1), dtype='float32')
time_axis[:, 0] = np.array(range(nt)) * dt
xline_axis[:, 0] = np.array(range(xlines)) + 1
iline_axis[:, 0] = np.array(range(ilines)) + 1
iline_show_ID = 1 - 1
xline_show_ID = xlines - 1

extent_iline = np.min(xline_axis), np.max(xline_axis), np.max(time_axis), np.min(time_axis)
extent_xline = np.min(iline_axis), np.max(iline_axis), np.max(time_axis), np.min(time_axis)
print(extent_iline)
print(extent_xline)
alpha_X = 1.0
alpha_Y = 0.5
labelpad_time_iline = -1
labelpad_time_xline = -1
labelpad_ixline = 0.7
colorbar_pad = 0.0072
font_normal = {'family': 'Arial', 'weight': 'normal', 'size': 8}
xline_ticks = np.hstack((1, (np.linspace(150, 600, num=int((600 - 150) / 150 + 1), endpoint=True)).astype(int), xlines))
iline_ticks = np.hstack((1, (np.linspace(100, 500, num=int((500 - 100) / 100 + 1), endpoint=True)).astype(int), ilines))
print(xline_ticks)
print(iline_ticks)
time_ticks = np.linspace(0, 5, num=int((5 - 0) / 0.2 + 1), endpoint=True)
cmapX_holdon = 'gray'
cmapX = we_seismic_cmap()
cmapY = mpl.colors.ListedColormap(['red', 'yellow', 'green', 'blue', 'fuchsia', 'brown'])
dpi = 600
axis_linewith = 0.3
fig_width = 3.33 * 0.59
fig_height = 3.999
position_fig_h = 100
position_fig_v = 50
d_fig_h = 230
d_fig_v = 400
wm_geometry_1 = '+' + str(position_fig_h + 0 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_2 = '+' + str(position_fig_h + 1 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_3 = '+' + str(position_fig_h + 2 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_4 = '+' + str(position_fig_h + 3 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_1)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(X[:, :, iline_show_ID].reshape(nt, xlines), cmap=cmapX, norm=normX, aspect='auto', alpha=alpha_X,
           origin='upper', extent=extent_iline)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Crossline number', font=font_normal, labelpad=labelpad_ixline)
plt.ylabel('Time (s)', font=font_normal, labelpad=labelpad_time_iline)
plt.xticks(xline_ticks, font=font_normal)
plt.yticks(time_ticks, font=font_normal)
plt.xlim(np.min(xline_axis) - 1, np.max(xline_axis) + 1)
plt.ylim(np.max(time_axis) + 1 * dt, np.min(time_axis) - 1 * dt)
colorbar_X(orientation='horizontal', font=font_normal, pad=colorbar_pad)
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)
plt.tight_layout()
if figure_save:
    fname = paper_figure_dir + '/' + 'FigDemoIlineX.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_2)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(X[:, :, iline_show_ID].reshape(nt, xlines), cmap=cmapX_holdon, norm=normX, aspect='auto', alpha=alpha_X,
           origin='upper', extent=extent_iline)
plt.imshow(Y[:, :, iline_show_ID].reshape(nt, xlines), cmap=cmapY, norm=normY, aspect='auto', alpha=alpha_Y,
           origin='upper', extent=extent_iline)
plt.text(150, 2.80, R[0], rotation=0, fontdict=font_normal_R, color='white')
plt.text(150, 2.45, R[1], rotation=330, fontdict=font_normal_R, color='black')
plt.text(150, 1.00, R[1], rotation=350, fontdict=font_normal_R, color='black')
plt.text(100, 2.20, R[2], rotation=335, fontdict=font_normal_R, color='white')
plt.text(100, 1.52, R[3], rotation=0, fontdict=font_normal_R, color='white')
plt.text(100, 0.20, R[3], rotation=0, fontdict=font_normal_R, color='white')
plt.text(150, 1.28, R[4], rotation=355, fontdict=font_normal_R, color='black')
plt.text(0.0, 0.72, R[5], rotation=350, fontdict=font_normal_R, color='white')
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
    fname = paper_figure_dir + '/' + 'FigDemoIlineXYHoldOn.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_3)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(X[:, xline_show_ID, :].reshape(nt, ilines), cmap=cmapX, norm=normX, aspect='auto', alpha=alpha_X,
           origin='upper', extent=extent_xline)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Inline number', font=font_normal, labelpad=labelpad_ixline)
plt.ylabel('Time (s)', font=font_normal, labelpad=labelpad_time_xline)
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
    fname = paper_figure_dir + '/' + 'FigDemoXlineX.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_4)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(X[:, xline_show_ID, :].reshape(nt, ilines), cmap=cmapX_holdon, norm=normX, aspect='auto', alpha=alpha_X,
           origin='upper', extent=extent_xline)
plt.imshow(Y[:, xline_show_ID, :].reshape(nt, ilines), cmap=cmapY, norm=normY, aspect='auto', alpha=alpha_Y,
           origin='upper', extent=extent_xline)
plt.text(100, 2.98, R[0], rotation=0, fontdict=font_normal_R, color='white')
plt.text(100, 2.68, R[1], rotation=35, fontdict=font_normal_R, color='black')
plt.text(100, 1.90, R[3], rotation=0, fontdict=font_normal_R, color='white')
plt.text(100, 0.40, R[3], rotation=0, fontdict=font_normal_R, color='white')
plt.text(0, 1.48, R[4], rotation=325, fontdict=font_normal_R, color='black')
plt.text(0.0, 1.20, R[5], rotation=345, fontdict=font_normal_R, color='white')
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
    fname = paper_figure_dir + '/' + 'FigDemoXlineXYHoldOn.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

plt.show()
