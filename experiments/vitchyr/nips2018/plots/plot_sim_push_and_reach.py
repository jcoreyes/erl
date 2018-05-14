import matplotlib.pyplot as plt
from railrl.misc import plot_util as plot
from railrl.misc.data_processing import get_trials
import numpy as np

from railrl.misc.plot_util import plot_trials

plt.style.use("ggplot")

# dirs = [
#     '/home/vitchyr/git/railrl/data/doodads3/05-12-sawyer-push-and-reach-easy/',
# ]
# f = plot.filter_by_flat_params({
#     'replay_buffer_kwargs.fraction_goals_are_env_goals': 0.5,
#     'replay_buffer_kwargs.fraction_goals_are_rollout_goals': 0.2,
#     'exploration_type': 'ou'
# })
# exps = plot.load_exps(dirs, f, suppress_output=False)
#
# dirs = [
#     '/home/vitchyr/git/railrl/data/doodads3/05-12-sawyer-push-and-reach-easy-vae-rl',
# ]
# f = plot.filter_by_flat_params({
#     'replay_kwargs.fraction_goals_are_env_goals': 0.5,
#     'replay_kwargs.fraction_goals_are_rollout_goals': 0.2,
# })
# exps += plot.load_exps(dirs, f, suppress_output=False)
# plot.split(exps, ["Final  puck_distance Mean"],
#            [],
#            default_vary={"do_state_based_exp": False},
#            smooth=plot.ma_filter(10),
#            print_final=False, print_min=False, print_plot=True)
# plt.title("Pusher2D, Distance to Goal")
# plt.show()


state_trials = get_trials(
    '/home/vitchyr/git/railrl/data/doodads3/05-12-sawyer-push-and-reach-easy/',
    criteria={
        'replay_buffer_kwargs.fraction_goals_are_env_goals': 0.5,
        'replay_buffer_kwargs.fraction_goals_are_rollout_goals': 0.2,
        'exploration_type': 'ou'
    }
)
my_trials = get_trials(
    '/home/vitchyr/git/railrl/data/doodads3/05-12-sawyer-push-and-reach-easy-vae-rl',
    criteria={
        'replay_kwargs.fraction_goals_are_env_goals': 0.5,
        'replay_kwargs.fraction_goals_are_rollout_goals': 0.2,
    }
)


# def y_process(y_keys, y_process):

y_keys = [
    'Final  puck_distance Mean',
    'Final  hand_distance Mean',
]
plot_trials(
    {
        'State': state_trials,
        'VAE': my_trials
    },
    y_keys=y_keys,
    # x_key=x_key,
)

plt.xlabel('Number of Environment Steps Total')
plt.ylabel('Final distance to Goal')
plt.savefig('/home/vitchyr/git/railrl/experiments/vitchyr/nips2018/plots'
            '/push_and_reach.jpg')
plt.show()

# plt.savefig("/home/ashvin/data/s3doodad/media/plots/pusher2d.pdf")
