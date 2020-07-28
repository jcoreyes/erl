import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

from railrl.misc.data_processing import get_trials

from common import plot_variant

plt.style.use("ggplot")

base_path = "/home/soroush/data/local/"

disco = get_trials(
    # osp.join(base_path, 'pb-4obj-rel/07-18-distr-inferred-n-30'),
    osp.join(base_path, 'pb-4obj-rel/07-28-eval-disco'),
)
gcrl = get_trials(
    # osp.join(base_path, 'pb-4obj-rel/07-19-point'),
    osp.join(base_path, 'pb-4obj-rel/07-28-eval-gcrl-oracle'),
)

name_to_trials = OrderedDict()
name_to_trials['DisCo RL (ours)'] = disco
name_to_trials['GCRL'] = gcrl
x_label = 'Num Env Steps Total (x1000)'
x_key = 'epoch'
x_lim = (0, 2000)
y_lim = (-0.1, 1.2)

### xy distance ###
y_keys = [
    # 'evalenv_infosfinalbowl_cube_0_dist_Mean',
    'evalenv_infosfinalbowl_cube_0_success Mean',
]
y_label = 'Success Rate'
plot_name = 'bullet_rel.pdf'
plot_variant(
    name_to_trials,
    plot_name,
    x_key, y_keys,
    x_label, y_label,
    x_lim=x_lim, y_lim=y_lim,
    show_legend=True,
    filter_frame=100,
    upper_limit=4.0,
    title='Sawyer Pick and Place',
)