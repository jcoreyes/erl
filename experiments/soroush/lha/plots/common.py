import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from railrl.visualization.plot_util import plot_trials
from railrl.visualization.plot_util import ma_filter

def plot_variant(
        name_to_trials,
        plot_name,
        x_key,
        y_keys,
        x_label,
        y_label,
        x_lim=None,
        y_lim=None,
        show_legend=True,
        filter_frame=10,
        upper_limit=None,
        title=None,
):
    plot_trials(
        name_to_trials,
        x_key=x_key,
        y_keys=y_keys,
        process_time_series=ma_filter(filter_frame),
    )

    if upper_limit is not None:
        plt.axhline(y=upper_limit, color='gray', linestyle='dashed')

    if show_legend:
        plt.legend()
    plt.xlabel(x_label)
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    plot_dir = '/home/soroush/research/railrl/experiments/soroush/lha/plots'
    full_plot_name = osp.join(plot_dir, plot_name)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(6.0, 3.0)
    fig.savefig(full_plot_name, bbox_inches='tight')

    # plt.savefig(full_plot_name, bbox_inches='tight')
    plt.show()
    plt.close()