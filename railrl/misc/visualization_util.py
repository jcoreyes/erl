import tempfile
import scipy.misc
from collections import namedtuple
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# The first dimension of values correspond to the x axis
HeatMap = namedtuple("HeatMap", ['values', 'x_values', 'y_values', 'info'])
VectorField = namedtuple("VectorField",
                         ['values', 'dx_values', 'dy_values', 'x_values',
                          'y_values', 'info'])


def make_heat_map(eval_func, x_bounds, y_bounds, *, resolution=10, info=None):
    """
    :param eval_func: eval_func(x, y) -> value
    :param x_bounds: 
    :param y_bounds: 
    :param resolution: 
    :param info: A dictionary to save inside the vector field
    :return: 
    """
    if info is None:
        info = {}
    x_values = np.linspace(*x_bounds, num=resolution)
    y_values = np.linspace(*y_bounds, num=resolution)
    map = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            map[i, j] = eval_func(x_values[i], y_values[j])
    return HeatMap(map, x_values, y_values, info)


def make_vector_field(eval_func, x_bounds, y_bounds, *, resolution=10,
                      info=None):
    """
    :param eval_func: eval_func(x, y) -> value, dx, dy
    :param x_bounds: 
    :param y_bounds: 
    :param resolution: 
    :param info: A dictionary to save inside the vector field
    :return: 
    """
    if info is None:
        info = {}
    x_values = np.linspace(*x_bounds, num=resolution)
    y_values = np.linspace(*y_bounds, num=resolution)
    values = np.zeros((resolution, resolution))
    dx_values = np.zeros((resolution, resolution))
    dy_values = np.zeros((resolution, resolution))

    for x in range(resolution):
        for y in range(resolution):
            value, dx, dy = eval_func(x_values[x], y_values[y])
            values[x, y] = value
            dx_values[x, y] = dx
            dy_values[x, y] = dy
    return VectorField(
        values=values,
        dx_values=dx_values,
        dy_values=dy_values,
        x_values=x_values,
        y_values=y_values,
        info=info,
    )


def plot_heatmap(fig, ax, heatmap):
    p, x, y, _ = heatmap
    im = ax.imshow(
        np.swapaxes(p, 0, 1),  # imshow uses first axis as y-axis
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap=plt.get_cmap('plasma'),
        interpolation='nearest',
        aspect='auto',
        origin='bottom',  # <-- Important! By default top left is (0, 0)
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    return im


def plot_vector_field(fig, ax, vector_field, skip_rate=1):
    skip = (slice(None, None, skip_rate), slice(None, None, skip_rate))
    p, dx, dy, x, y, _ = vector_field
    im = ax.imshow(
        np.swapaxes(p, 0, 1),  # imshow uses first axis as y-axis
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap=plt.get_cmap('plasma'),
        interpolation='nearest',
        aspect='auto',
        origin='bottom',  # <-- Important! By default top left is (0, 0)
    )
    x, y = np.meshgrid(x, y)
    ax.quiver(x[skip], y[skip], dx[skip], dy[skip])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')


def save_image(fig=None, fname=None):
    if fname is None:
        fname = tempfile.TemporaryFile()
    if fig is not None:
        fig.savefig(fname)
    else:
        plt.savefig(fname, format='png')
    plt.close('all')
    fname.seek(0)
    img = scipy.misc.imread(fname)
    fname.close()
    return img
