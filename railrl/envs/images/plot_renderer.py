import collections
import numpy as np
from railrl.envs.images import Renderer
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt


class ScrollingPlotRenderer(Renderer):
    """
    Plot the history of some number over a scrolling window.

    So something like:

    ---------------------
    |                   |
    |   |        ___/   |
    |   |     __/       |
    | y |    /          |
    |   | __/           |
    |   |_____________  |
    |        time       |
    ---------------------
    """
    def __init__(
            self,
            min_value=None,
            max_value=None,
            window_size=10,
            dpi=32,
            **kwargs
    ):
        """Render an image."""
        super().__init__(create_image_format='HWC', **kwargs)
        self._min_value = min_value # TODO: use
        self._max_value = max_value
        self._dpi = dpi # TODO: use automatically
        _, height, width = self.image_chw

        figsize = (height / dpi, width / dpi)
        self.fig = plt.figure(
            figsize=figsize,
            dpi=dpi,
        )
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self.fig)
        self.xs = collections.deque(maxlen=window_size)
        self.ys = collections.deque(maxlen=window_size)
        self.lines = self.ax.plot(
            [], [],
        )[0]
        self.t = 0

    def reset(self):
        self.t = 0

    def _create_image(self, number):
        self._update_plot(number)
        return self._render_plot()

    def _update_plot(self, number):
        self.xs.append(self.t)
        self.t += 1
        self.ys.append(number)
        self.lines.set_xdata(np.array(self.xs))
        self.lines.set_ydata(np.array(self.ys))
        self.ax.relim()
        self.ax.autoscale_view()

    def _render_plot(self):
        # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
        self.canvas.draw()
        flat_img = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8')
        width, height = self.fig.get_size_inches() * self.fig.get_dpi()
        width = int(width)
        height = int(height)
        img = flat_img.reshape(height, width, 3)
        return img