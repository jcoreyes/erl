import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

from rlkit.misc.data_processing import get_trials

from common import plot_variant

plt.style.use("ggplot")

base_path = "/home/soroush/data/local/"

disco_off_policy = get_trials(
    osp.join(base_path, 'pg-4obj-maskgen/07-28-eval-disco-off-policy'),
)
disco_on_policy = get_trials(
    osp.join(base_path, 'pg-4obj-maskgen/07-28-eval-disco-on-policy'),
)

name_to_trials = OrderedDict()
name_to_trials['Off policy'] = disco_off_policy
name_to_trials['On policy'] = disco_on_policy
x_label = 'Num Env Steps Total (x1000)'
x_key = 'epoch'
x_lim = (0, 3000)
y_lim = (0, 2.05)

### xy distance ###
y_keys = [
    'evalenv_infosfinalnum_obj_1_2_success Mean',
]
y_label = 'Num Successful Steps'
plot_name = 'pg_maskgen.pdf'
plot_variant(
    name_to_trials,
    plot_name,
    x_key, y_keys,
    x_label, y_label,
    x_lim=x_lim, y_lim=y_lim,
    show_legend=True,
    filter_frame=100,
    upper_limit=2.0,
    # title='2D Pick and Place',
)