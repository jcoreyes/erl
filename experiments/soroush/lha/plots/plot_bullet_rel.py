import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

from rlkit.misc.data_processing import get_trials

from common import plot_variant

plt.style.use("ggplot")
base_path = "/home/soroush/data/local/"

disco = get_trials(
    osp.join(base_path, 'pb-4obj-rel/07-28-eval-disco'),
    excluded_seeds=[98106],
)
gcrl = get_trials(
    osp.join(base_path, 'pb-4obj-rel/07-28-eval-gcrl-oracle'),
)
vice = get_trials(
    osp.join(base_path, 'pb-4obj-rel/pb-4obj-rel-vice'),
)
disco_hard_coded = get_trials(
    osp.join(base_path, 'pb-4obj-rel/07-28-eval-disco-hard-coded'),
)
sac = get_trials(
    osp.join(base_path, 'pb-4obj-rel/07-28-eval-vanilla-rl'),
)

name_to_trials = OrderedDict()
name_to_trials['DisCo RL (ours)'] = disco
name_to_trials['GCRL'] = gcrl
name_to_trials['VICE'] = vice
name_to_trials['DisCo RL + hard-coded $\omega$'] = disco_hard_coded
name_to_trials['SAC'] = sac
x_label = 'Num Env Steps Total (x1000)'
x_key = 'epoch'
x_lim = (0, 2000)
y_lim = (-0.1, 1.05)

### xy distance ###
y_keys = [
    'evaluationenv_infosfinalbowl_cube_0_success Mean',
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
    upper_limit=1.0,
    # title='Sawyer Pick and Place: Single Task',
)