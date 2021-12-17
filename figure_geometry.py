import matplotlib as mpl
from pyseismic_local.plot_functions import *
import os

paper_figure_dir = './' + 'paper_figure'
if not os.path.exists(paper_figure_dir):
    os.mkdir(paper_figure_dir)

figure_save = 1

ilines_training = 590
xlines_training = 782

ilines_test1 = 251
xlines_test1 = 782

ilines_test2 = ilines_training + ilines_test1
xlines_test2 = 334

ilines_all = ilines_test2
xlines_all = xlines_test1 + xlines_test2

GeoMetry = np.zeros(shape=(xlines_all, ilines_all), dtype='float32')
print(f'GeoMetry.shape: {GeoMetry.shape}')
# training
left_for_test_iline = 7
left_for_test_xline = 45
GeoMetry[left_for_test_xline:xlines_training - left_for_test_xline, left_for_test_iline:ilines_training - left_for_test_iline] = 1
# TestData_Image1
GeoMetry[0:xlines_training, ilines_training:ilines_all] = 2
# TestData_Image2
GeoMetry[xlines_training:xlines_all, 0:ilines_all] = 3
print(GeoMetry)

# fig = plt.figure()
# plt.plot(GeoMetry[left_for_test_xline,:])

print(f'{100 * (xlines_training - 2 * left_for_test_xline) * (ilines_training - 2 * left_for_test_iline) / (782 * 590)}')
print(f'{100 * ((782 * 590) - (xlines_training - 2 * left_for_test_xline) * (ilines_training - 2 * left_for_test_iline)) / (782 * 590)}')

print(f'{100 * (xlines_training - 2 * left_for_test_xline) * (ilines_training - 2 * left_for_test_iline) / (1116 * 841)}')



iline_axis = np.zeros(shape=(GeoMetry.shape[1], 1), dtype='int')
xline_axis = np.zeros(shape=(GeoMetry.shape[0], 1), dtype='int')
iline_axis[:, 0] = np.array(range(GeoMetry.shape[1])) + 1
xline_axis[:, 0] = np.array(range(GeoMetry.shape[0])) + 1

axis_linewith = 0.1

norm = mpl.colors.Normalize(vmin=np.min(GeoMetry), vmax=np.max(GeoMetry))
extent = np.max(iline_axis), np.min(iline_axis), np.max(xline_axis), np.min(xline_axis)

labelpad_iline = -1
labelpad_xline = -6
dpi = 600
font_normal = {'family': 'Arial', 'weight': 'normal', 'size': 8, }
fig = plt.figure()
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+700+100")
fig.set_size_inches(3.33 * 1.25, 5.0)
colors = ['red', 'blue', 'green', 'yellow']
cmap = mpl.colors.ListedColormap(colors)
plt.imshow(np.fliplr(GeoMetry), cmap=cmap, norm=norm, aspect='equal', origin='upper', extent=extent)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Inline number', font=font_normal, labelpad=labelpad_iline)
plt.ylabel('Crossline number', font=font_normal, labelpad=labelpad_xline)
plt.xticks([1, ilines_training, ilines_all], font=font_normal)
plt.yticks([1, 46, 737, xlines_training, xlines_all], font=font_normal)
plt.xlim(np.max(iline_axis) + 1, np.min(iline_axis) - 1)
plt.ylim(np.max(xline_axis) + 1, np.min(xline_axis) - 1)
plt.gca().spines['bottom'].set_linewidth(axis_linewith)
plt.gca().spines['left'].set_linewidth(axis_linewith)
plt.gca().spines['top'].set_linewidth(axis_linewith)
plt.gca().spines['right'].set_linewidth(axis_linewith)

fname = paper_figure_dir + '/' + 'FigGeometry.tif'
if figure_save:
    plt.savefig(fname=fname, dpi=dpi, facecolor='w', edgecolor='w')

plt.show()
