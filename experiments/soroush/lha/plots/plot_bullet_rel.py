import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

from railrl.misc.data_processing import get_trials
from railrl.visualization.plot_util import plot_trials
from railrl.visualization.plot_util import ma_filter

plt.style.use("ggplot")

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
):
    plot_trials(
        name_to_trials,
        x_key=x_key,
        y_keys=y_keys,
        process_time_series=ma_filter(100),
    )
    if show_legend:
        plt.legend()
    plt.xlabel(x_label)
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.ylabel(y_label)
    plt.title("Sawyer Pick and Place")
    plot_dir = '/home/soroush/research/railrl/experiments/soroush/lha/plots'
    full_plot_name = osp.join(plot_dir, plot_name)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(6.0, 3.0)
    fig.savefig(full_plot_name, bbox_inches='tight')

    # plt.savefig(full_plot_name, bbox_inches='tight')
    plt.show()
    plt.close()

base_path = "/home/soroush/data/local/"

gdcrl = get_trials(
    osp.join(base_path, 'pb-4obj-rel/07-18-distr-inferred-n-30'),
)
gcrl = get_trials(
    osp.join(base_path, 'pb-4obj-rel/07-19-point'),
)

name_to_trials = OrderedDict()
name_to_trials['DisCo RL (ours)'] = gdcrl
name_to_trials['GCRL'] = gcrl
    # 'GCRL': gcrl,
    # 'Ours - mean relabeling': gdcrl_no_goal_relabeling,
    # 'Ours - cov relabeling': gdcrl_no_mask_relabeling,
# )
x_label = 'Number of Environment Steps Total (x1000)'
x_key = 'epoch'
x_lim = (0, 2500)

### xy distance ###
y_keys = [
    'evalenv_infosfinalbowl_cube_0_dist_Mean',
    # 'epoch'
]
y_label = 'Final Relative Distance'
plot_name = 'bullet_rel.pdf'
plot_variant(
    name_to_trials,
    plot_name,
    x_key, y_keys,
    x_label, y_label,
    x_lim=x_lim,
    show_legend=True,
)