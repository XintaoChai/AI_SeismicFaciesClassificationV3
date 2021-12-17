import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import ListedColormap


def colorbar_X(orientation, font, pad):
    colorbar = plt.colorbar(orientation=orientation, pad=pad)
    for l in colorbar.ax.xaxis.get_ticklabels():
        l.set_family(font['family'])
        l.set_size(font['size'])


def colorbar_Y(ncolors, cmap, orientation, tick, font, pad):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(0.5, ncolors + 0.5)
    colorbar = plt.colorbar(mappable, orientation=orientation, pad=pad)
    colorbar.set_ticks(np.linspace(1, ncolors, ncolors))
    colorbar.set_ticklabels(tick)
    for l in colorbar.ax.xaxis.get_ticklabels():
        l.set_family(font['family'])
        l.set_size(font['size'])


def cmap_discretize(cmap, N):
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)]
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def we_seismic_cmap():
    taper = np.linspace(0, 1, 128)
    taper2 = np.concatenate((np.zeros(64), np.linspace(0, 1, 64)))
    r = np.concatenate((taper, np.ones(128)))
    g = np.concatenate((taper, taper[::-1]))
    b = np.concatenate((taper, taper2[::-1]))
    r = np.expand_dims(r, axis=1)
    g = np.expand_dims(g, axis=1)
    b = np.expand_dims(b, axis=1)
    rgb = np.concatenate((r, g, b), axis=1)
    colormap = ListedColormap(rgb)
    return colormap
