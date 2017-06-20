from collections import OrderedDict
from numbers import Number

import numpy as np
import os


def create_stats_ordered_dict(name, data, stat_prefix=None):
    if stat_prefix is not None:
        name = "{} {}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, np.ndarray) and data.size == 1:
        return OrderedDict({name: float(data)})

    return OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
        (name + ' Max', np.max(data)),
        (name + ' Min', np.min(data)),
    ])


def get_dirs(root):
    """
    Get a list of all the directories under this directory.
    """
    for root, directories, filenames in os.walk(root):
        for directory in directories:
            yield os.path.join(root, directory)


def make_heat_map(eval_func, *, resolution=50, min_val=-1, max_val=1):
    linspace = np.linspace(min_val, max_val, num=resolution)
    map = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            map[i, j] = eval_func(np.array([linspace[i], linspace[j]]))
    return map


def make_density_map(paths, *, resolution=50, min_val=-1, max_val=1):
    linspace = np.linspace(min_val, max_val, num=resolution)
    y = paths[:, 0]
    x = paths[:, 1]
    H, xedges, yedges = np.histogram2d(y, x, bins=(linspace, linspace))
    H = H.astype(np.float)
    H = H / np.max(H)
    return H


def plot_maps(old_combined=None, *heatmaps):
    import matplotlib.pyplot as plt
    combined = np.c_[heatmaps]
    if old_combined is not None:
        combined = np.r_[old_combined, combined]
    plt.figure()
    plt.imshow(combined, cmap='afmhot', interpolation='none')
    plt.show()
    return combined


if __name__ == "__main__":
    def evalfn(a):
        return np.linalg.norm(a)

    hm = make_heat_map(evalfn, resolution=50)
    paths = np.random.randn(5000, 2) * 0.1
    dm = make_density_map(paths, resolution=50)
    a = plot_maps(None, hm, dm)
    plot_maps(a, hm, dm)
