import matplotlib as mpl
import segyio
import matplotlib.colors as mcolors
from pyseismic_local.plot_functions import *
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

X = segyio.tools.cube(raw_data_X_ID[0])
Y = segyio.tools.cube(raw_data_Y_ID[0])
X = X.transpose((2, 1, 0)).astype('float32')
Y = Y.transpose((2, 1, 0)).astype('float32')
print('Y.shape:' + str(Y.shape))
patch_size_height = Y.shape[0]
patch_size_width = Y.shape[1]

figure_save = 1
dpi = 300

zoom_X = 0.1
X_max = 5522.086
X_min = -5195.5234

normY = mpl.colors.Normalize(vmin=1, vmax=6)
normX = mpl.colors.Normalize(vmin=X_min * zoom_X, vmax=X_max * zoom_X)

axis_linewith = 0
dt = 1
nt = X.shape[0]
xlines = X.shape[1]
ilines = X.shape[2]
time_axis = np.zeros(shape=(nt, 1), dtype='float32')
xline_axis = np.zeros(shape=(xlines, 1), dtype='float32')
iline_axis = np.zeros(shape=(ilines, 1), dtype='float32')
time_axis[:, 0] = (np.array(range(nt)) + 1) * dt
xline_axis[:, 0] = np.array(range(xlines)) + 1
iline_axis[:, 0] = np.array(range(ilines)) + 1
iline_show_ID = 1 - 1

extent_iline = np.min(xline_axis), np.max(xline_axis), np.max(time_axis), np.min(time_axis)
extent_xline = np.min(iline_axis), np.max(iline_axis), np.max(time_axis), np.min(time_axis)
print(extent_iline)
print(extent_xline)
alpha_X = 1.0
alpha_Y = 1.0
font_normal = {'family': 'Arial', 'weight': 'normal', 'size': 8}
xline_ticks = np.hstack((1, (np.linspace(100, 600, num=int((600 - 100) / 100 + 1), endpoint=True)).astype(int), xlines))
iline_ticks = np.hstack((1, (np.linspace(100, 500, num=int((500 - 100) / 100 + 1), endpoint=True)).astype(int), ilines))
print(xline_ticks)
print(iline_ticks)
time_ticks = np.linspace(0, 5, num=int((5 - 0) / 0.1 + 1), endpoint=True)
cmapX = we_seismic_cmap()
cmapY = mpl.colors.ListedColormap(['red', 'yellow', 'green', 'blue', 'fuchsia', 'brown'])

fig_width = 3.33 * 0.409
fig_height = 4.81
position_fig_h = 100
position_fig_v = 50
d_fig_h = 390
d_fig_v = 400
wm_geometry_1 = '+' + str(position_fig_h + 0 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
wm_geometry_2 = '+' + str(position_fig_h + 1 * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_1)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(X[:, :, iline_show_ID].reshape(nt, xlines), cmap=cmapX, norm=normX, aspect='equal', alpha=alpha_X, origin='upper', extent=extent_iline)
plt.xlim(1, 576)
plt.ylim(992, 1)
plt.xticks([])
plt.yticks([])
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)

plt.tight_layout()
if figure_save:
    fname = paper_figure_dir + '/' + 'FigTrainX.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry(wm_geometry_2)
fig.set_size_inches(fig_width, fig_height)
plt.imshow(Y[:, :, iline_show_ID].reshape(nt, xlines), cmap=cmapY, norm=normY, aspect='equal', alpha=alpha_Y, origin='upper', extent=extent_iline)
plt.xlim(1, 576)
plt.ylim(992, 1)
plt.xticks([])
plt.yticks([])
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)

plt.tight_layout()
if figure_save:
    fname = paper_figure_dir + '/' + 'FigTrainY.tif'
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

plt.show()
