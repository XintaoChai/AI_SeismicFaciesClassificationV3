import matplotlib as mpl
import matplotlib.colors as mcolors
from pyseismic_local.plot_functions import *
from pyseismic_local.data_read_write_functions import *
from pyseismic_local.config import *
################################################################################
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
################################################################################
paper_figure_dir = './' + 'paper_figure'
if not os.path.exists(paper_figure_dir):
    os.mkdir(paper_figure_dir)

    ############################################################################
training_data_dir = module_dir + '/' + 'training_data_992x576_stride45_5_Ychannels1'
X_channels = 1
Y_channels = 1
available_demo_number = 10048
actual_demo_number = 4
training_indexes = np.array([1, 6960, 6961, 10048])
training_indexes = training_indexes - 1

patch_rows = 992
patch_cols = 576
demoX = np.zeros(shape=[actual_demo_number, patch_rows, patch_cols, X_channels], dtype='float32')
demoY = np.zeros(shape=[actual_demo_number, patch_rows, patch_cols, Y_channels], dtype='float32')

print(training_indexes)
X_normal = 5522.086
X_max = 5522.086 / X_normal
X_min = -5195.5234 / X_normal

for j in range(actual_demo_number):
    demoX_dataID = training_data_dir + '/' + format(training_indexes[j] + 1, '011d') + 'X.bin'
    demoY_dataID = training_data_dir + '/' + format(training_indexes[j] + 1, '011d') + 'Y.bin'
    demoX_data = bin_data_reader2D_float32(demoX_dataID, patch_rows, patch_cols, X_channels, 1)
    demoY_data = bin_data_reader2D_float32(demoY_dataID, patch_rows, patch_cols, Y_channels, 1)
    demoX[j] = demoX_data[0]
    demoY[j] = demoY_data[0]
    del demoX_dataID, demoX_data, demoY_dataID, demoY_data

demoX = demoX / X_normal

zoom_X = 0.1

normY = mpl.colors.Normalize(vmin=1, vmax=6)
normX = mpl.colors.Normalize(vmin=X_min * zoom_X, vmax=X_max * zoom_X)

axis_linewith = 0.3
vert_axis = np.zeros(shape=(patch_rows, 1), dtype='int32')
hori_axis = np.zeros(shape=(patch_cols, 1), dtype='int32')
vert_axis[:, 0] = np.array(range(patch_rows)) + 1
hori_axis[:, 0] = np.array(range(patch_cols)) + 1

extent_demo = np.min(hori_axis), np.max(hori_axis), np.max(vert_axis), np.min(vert_axis)
print(extent_demo)
alpha_X = 1.0
alpha_Y = 1.0
font_normal = {'family': 'Arial', 'weight': 'normal', 'size': 8}
font_normal_R = {'family': 'Arial', 'weight': 'normal', 'size': 7}
hori_ticks = np.hstack(
    (1, (np.linspace(100, 400, num=int((400 - 100) / 100 + 1), endpoint=True)).astype(int), patch_cols))
print(hori_ticks)
vert_ticks = np.hstack(
    (1, (np.linspace(100, 900, num=int((900 - 100) / 100 + 1), endpoint=True)).astype(int), patch_rows))
cmapY = mpl.colors.ListedColormap(['red', 'yellow', 'green', 'blue', 'fuchsia', 'brown'])

figure_save = 1
fig_width = 3.33 * 0.59
fig_height = 3.999
position_fig_h = 200
position_fig_v = 3
d_fig_h = 270
d_fig_v = 500
dpi = 600

labelpad_vert = -1
labelpad_hori = 0.7
colorbar_pad = 0.0072
cmapX = we_seismic_cmap()

R = ["Basement/other", "Slope mudstone A", "Mass transport deposit", "Slope mudstone B", "Slope valley",
     "Submarine canyon system"]

for i_demo in range(actual_demo_number):
    wm_geometry_1 = '+' + str(position_fig_h + i_demo * d_fig_h) + '+' + str(position_fig_v + 0 * d_fig_v)
    wm_geometry_2 = '+' + str(position_fig_h + i_demo * d_fig_h) + '+' + str(position_fig_v + 1 * d_fig_v)
    fig = plt.figure()
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry(wm_geometry_1)
    fig.set_size_inches(fig_width, fig_height)
    plt.imshow(demoX[i_demo].reshape(patch_rows, patch_cols), cmap=cmapX, norm=normX, aspect='auto', alpha=alpha_X,
               origin='upper', extent=extent_demo)

    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')
    plt.xlim(0, patch_cols + 1)
    plt.ylim(patch_rows + 1, 0)
    colorbar_X(orientation='horizontal', font=font_normal, pad=colorbar_pad)
    plt.xlabel('Number of points', font=font_normal, labelpad=labelpad_hori)
    plt.ylabel('Number of points', font=font_normal, labelpad=labelpad_vert)
    plt.xticks(hori_ticks, font=font_normal)
    plt.yticks(vert_ticks, font=font_normal)
    plt.gca().spines['bottom'].set_linewidth(axis_linewith)
    plt.gca().spines['left'].set_linewidth(axis_linewith)
    plt.gca().spines['top'].set_linewidth(axis_linewith)
    plt.gca().spines['right'].set_linewidth(axis_linewith)

    plt.tight_layout()
    if figure_save:
        fname = paper_figure_dir + '/' + 'FigTrainX' + str(i_demo + 1) + '.tif'
        plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

    fig = plt.figure()
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry(wm_geometry_2)
    fig.set_size_inches(fig_width, fig_height)
    plt.imshow(demoY[i_demo].reshape(patch_rows, patch_cols), cmap=cmapY, norm=normY, aspect='auto', alpha=alpha_Y,
               origin='upper', extent=extent_demo)
    if i_demo == 0:
        plt.text(100, 2.80 * 992 / 3, R[0], rotation=0, fontdict=font_normal_R, color='white')
        plt.text(100, 2.40 * 992 / 3, R[1], rotation=340, fontdict=font_normal_R, color='black')
        plt.text(100, 1.00 * 992 / 3, R[1], rotation=350, fontdict=font_normal_R, color='black')
        plt.text(50, 2.15 * 992 / 3, R[2], rotation=340, fontdict=font_normal_R, color='white')
        plt.text(100, 1.52 * 992 / 3, R[3], rotation=0, fontdict=font_normal_R, color='white')
        plt.text(100, 0.20 * 992 / 3, R[3], rotation=0, fontdict=font_normal_R, color='white')
        plt.text(100, 1.28 * 992 / 3, R[4], rotation=355, fontdict=font_normal_R, color='black')
        plt.text(0, 0.63 * 992 / 3, R[5], rotation=355, fontdict=font_normal_R, color='white')
    elif i_demo == 1:
        plt.text(100, 2.80 * 992 / 3, R[0], rotation=0, fontdict=font_normal_R, color='white')
        plt.text(20, 700, R[1], rotation=0, fontdict=font_normal_R, color='black')
        plt.text(0, 370, R[1], rotation=345, fontdict=font_normal_R, color='black')
        plt.text(0, 448, R[3], rotation=0, fontdict=font_normal_R, color='white')
        plt.text(0, 625, R[2], rotation=0, fontdict=font_normal_R, color='black')
        plt.text(100, 100, R[3], rotation=0, fontdict=font_normal_R, color='white')
        plt.text(50, 350, R[5], rotation=335, fontdict=font_normal_R, color='white')
    elif i_demo == 2:
        plt.text(150, 2.80 * 992 / 3, R[0], rotation=0, fontdict=font_normal_R, color='white')
        plt.text(100, 720, R[1], rotation=0, fontdict=font_normal_R, color='black')
        plt.text(0, 330, R[1], rotation=0, fontdict=font_normal_R, color='black')
        plt.text(0, 625, R[2], rotation=0, fontdict=font_normal_R, color='black')
        plt.text(100, 100, R[3], rotation=0, fontdict=font_normal_R, color='white')
        # plt.text(0.0, 200, R[5], rotation=0, fontdict=font_normal_R, color='black')
    elif i_demo == 3:
        plt.text(100, 970, R[0], rotation=0, fontdict=font_normal_R, color='white')
        plt.text(50, 900, R[1], rotation=35, fontdict=font_normal_R, color='black')
        plt.text(0, 650, R[3], rotation=330, fontdict=font_normal_R, color='white')
        plt.text(100, 100, R[3], rotation=0, fontdict=font_normal_R, color='white')
        plt.text(0, 490, R[4], rotation=325, fontdict=font_normal_R, color='black')
        plt.text(0.0, 380, R[5], rotation=345, fontdict=font_normal_R, color='white')
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')
    plt.xlim(0, patch_cols + 1)
    plt.ylim(patch_rows + 1, 0)
    colorbar_Y(ncolors=6, cmap=cmapY, orientation='horizontal', tick=np.array(range(6)) + 1, font=font_normal,
               pad=colorbar_pad)
    plt.xlabel('Number of points', font=font_normal, labelpad=labelpad_hori)
    plt.ylabel('Number of points', font=font_normal, labelpad=labelpad_vert)
    plt.xticks(hori_ticks, font=font_normal)
    plt.yticks(vert_ticks, font=font_normal)
    plt.gca().spines['bottom'].set_linewidth(axis_linewith)
    plt.gca().spines['left'].set_linewidth(axis_linewith)
    plt.gca().spines['top'].set_linewidth(axis_linewith)
    plt.gca().spines['right'].set_linewidth(axis_linewith)

    plt.tight_layout()
    if figure_save:
        fname = paper_figure_dir + '/' + 'FigTrainY' + str(i_demo + 1) + '.tif'
        plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

plt.show()
