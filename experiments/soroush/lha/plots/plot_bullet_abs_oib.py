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
    # osp.join(base_path, 'pb-4obj/07-19-distr-inferred-n-30'),
    osp.join(base_path, 'pb-4obj/07-28-eval-disco-oib-postfix'),
)
gcrl = get_trials(
    osp.join(base_path, 'pb-4obj/07-28-eval-gcrl-oib-postfix'),
)
vice = get_trials(
    # osp.join(base_path, 'pb-4obj/pb-4obj-vice-oib'),
    osp.join(base_path, 'pb-4obj/pb-4obj-vice-oib-v2'),
)
disco_no_mean_rlbl = get_trials(
    osp.join(base_path, 'pb-4obj/07-28-eval-disco-no-mean-rlbl-oib-postfix'),
)
disco_no_cov_rlbl = get_trials(
    osp.join(base_path, 'pb-4obj/07-28-eval-disco-no-cov-rlbl-oib-postfix'),
)

name_to_trials = OrderedDict()
name_to_trials['DisCo RL (ours)'] = disco
name_to_trials['GCRL'] = gcrl
name_to_trials['VICE'] = vice
name_to_trials['Ours: no mean relabeling'] = disco_no_mean_rlbl
name_to_trials['Ours: no cov relabeling'] = disco_no_cov_rlbl
x_label = 'Num Env Steps Total (x1000)'
x_key = 'epoch'
x_lim = (0, 4000)
y_lim = (-0.1, 4.2)

del name_to_trials['Ours: no mean relabeling']
del name_to_trials['Ours: no cov relabeling']

y_keys = [
    'evaluationenv_infosfinalnum_bowl_obj_success Mean',
    'evalenv_infosfinalnum_bowl_obj_success Mean',
]
y_label = 'Num Successful Steps'
plot_name = 'bullet_abs_oib.pdf'
plot_variant(
    name_to_trials,
    plot_name,
    x_key, y_keys,
    x_label, y_label,
    x_lim=x_lim, y_lim=y_lim,
    # show_legend=True,
    filter_frame=100,
    upper_limit=4.0,
    title='Sawyer',
)

y_lim = (0.0, 0.75)
y_keys = [
    'evaluationenv_infosfinalhand_dist Mean',
    'evalenv_infosfinalhand_dist Mean',
]
y_label = 'Final End Effector Dist'
plot_name = 'bullet_abs_oib_ee.pdf'
plot_variant(
    name_to_trials,
    plot_name,
    x_key, y_keys,
    x_label, y_label,
    x_lim=x_lim, y_lim=y_lim,
    # show_legend=True,
    filter_frame=100,
    title='Sawyer',
)