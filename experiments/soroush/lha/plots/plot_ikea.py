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
    osp.join(base_path, 'shelf-4obj/07-28-eval-disco'),
)
gcrl = get_trials(
    osp.join(base_path, 'shelf-4obj/07-28-eval-gcrl'),
)
vice = get_trials(
    osp.join(base_path, 'shelf-4obj/shelf-4obj-vice'),
)

name_to_trials = OrderedDict()
name_to_trials['DisCo RL (ours)'] = disco
name_to_trials['GCRL'] = gcrl
name_to_trials['VICE'] = vice
x_label = 'Num Env Steps Total (x1000)'
x_key = 'epoch'
x_lim = (0, 2500)
y_lim = (-0.1, 4.05)

y_keys = [
    'evaluationenv_infosfinalnum_success_steps Mean',
    'evalenv_infosfinalnum_success_steps Mean',
]
y_label = 'Num Successful Steps'
plot_name = 'ikea.pdf'
plot_variant(
    name_to_trials,
    plot_name,
    x_key, y_keys,
    x_label, y_label,
    x_lim=x_lim, y_lim=y_lim,
    # show_legend=True,
    filter_frame=100,
    upper_limit=4.0,
    title='IKEA',
)

y_keys = [
    'evaluationenv_infosfinalcursor1_dist Mean',
    'evalenv_infosfinalcursor1_dist Mean',
]
y_label = 'Final End Effector Dist'
y_lim = (0, 1.0)
plot_name = 'ikea_ee.pdf'
plot_variant(
    name_to_trials,
    plot_name,
    x_key, y_keys,
    x_label, y_label,
    x_lim=x_lim, y_lim=y_lim,
    # show_legend=True,
    filter_frame=100,
    title='IKEA',
)