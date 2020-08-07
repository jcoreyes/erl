import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_util import plot_trials
from plot_util import ma_filter

def plot_variant(
        name_to_trials,
        plot_name,
        x_key,
        y_keys,
        x_label,
        y_label,
        x_lim=None,
        y_lim=None,
        show_legend=False,
        filter_frame=10,
        upper_limit=None,
        title=None,
):
    plt.rcParams.update({'font.size': 14})
    COLOR = 'black'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR

    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"

    plot_trials(
        name_to_trials,
        x_key=x_key,
        y_keys=y_keys,
        process_time_series=ma_filter(filter_frame),
    )

    if upper_limit is not None:
        plt.axhline(y=upper_limit, color='gray', linestyle='dashed')

    if show_legend:
        # plt.legend()
        plt.legend(bbox_to_anchor=(1.4, 0.5), loc='center', ncol=1)

        # ax = plt.gca()
        # leg = ax.get_legend()
        # leg.legendHandles[0].set_color('blue')
        # leg.legendHandles[1].set_color('orange')
        # leg.legendHandles[2].set_color('green')
        # leg.legendHandles[3].set_color('red')
        # leg.legendHandles[4].set_color('purple')

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