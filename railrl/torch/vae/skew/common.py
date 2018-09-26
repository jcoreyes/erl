import math
import numpy as np
from matplotlib import pyplot as plt

from railrl.misc import visualization_util as vu


class Dynamics(object):
    def __init__(self, projection, noise):
        self.projection = projection
        self.noise = noise

    def __call__(self, samples):
        new_samples = samples + self.noise * np.random.randn(
            *samples.shape
        )
        return self.projection(new_samples)


def plot_curves(names_and_data, report):
    n_curves = len(names_and_data)
    if n_curves < 4:
        n_cols = n_curves
        n_rows = 1
    else:
        n_cols = n_curves // 2
        n_rows = math.ceil(float(n_curves) / n_cols)

    plt.figure()
    for i, (name, data) in enumerate(names_and_data):
        j = i + 1
        plt.subplot(n_rows, n_cols, j)
        plt.plot(np.array(data))
        plt.title(name)
    fig = plt.gcf()
    img = vu.save_image(fig)
    report.add_image(img, "Final Distribution")