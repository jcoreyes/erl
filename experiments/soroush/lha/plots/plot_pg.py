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
    # osp.join(base_path, 'pg-4obj/07-21-distr-use-proper-mean-inferred-n-30'),
    osp.join(base_path, 'pg-4obj/07-28-eval-disco'),
)
gcrl = get_trials(
    # osp.join(base_path, 'pg-4obj/07-21-point'),
    osp.join(base_path, 'pg-4obj/07-28-eval-gcrl'),
)
gdcrl_no_mean_rlbl = get_trials(
    # osp.join(base_path, 'pg-4obj/07-21-inferred-n-30-no-goal-relabeling'),
    osp.join(base_path, 'pg-4obj/07-28-eval-disco-no-mean-rlbl'),
)
gdcrl_no_cov_rlbl = get_trials(
    # osp.join(base_path, 'pg-4obj/07-21-inferred-n-30-no-mask-relabeling'),
    osp.join(base_path, 'pg-4obj/07-28-eval-disco-no-cov-rlbl'),
)

name_to_trials = OrderedDict()
name_to_trials['DisCo RL (ours)'] = disco
name_to_trials['GCRL'] = gcrl
name_to_trials['Ours: no mean relabeling'] = gdcrl_no_mean_rlbl
name_to_trials['Ours: no cov relabeling'] = gdcrl_no_cov_rlbl
    # 'GCRL': gcrl,
    # 'Ours - mean relabeling': gdcrl_no_goal_relabeling,
    # 'Ours - cov relabeling': gdcrl_no_mask_relabeling,
# )
x_label = 'Num Env Steps Total (x1000)'
x_key = 'epoch'
x_lim = (0, 2000)
y_lim = (0, 4.5)

### xy distance ###
y_keys = [
    # 'evalenv_infosfinaldistance_to_target_obj_4 Mean',
    'evalenv_infosfinalnum_obj_success Mean',
]
y_label = 'Num Succesful Steps'
plot_name = 'pg.pdf'
plot_variant(
    name_to_trials,
    plot_name,
    x_key, y_keys,
    x_label, y_label,
    x_lim=x_lim, y_lim=y_lim,
    show_legend=True,
    filter_frame=100,
    upper_limit=4.0,
    title='2D Pick and Place',
)