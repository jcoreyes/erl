import matplotlib.pyplot as plt

from rlkit.misc.data_processing import get_trials
from rlkit.visualization.plot_util import plot_trials

plt.style.use("ggplot")



state_trials = get_trials(
    '/home/vitchyr/git/railrl/data/doodads3/05-12-sawyer-push-and-reach-easy/',
    criteria={
        'replay_buffer_kwargs.fraction_resampled_goals_are_env_goals': 0.5,
        'replay_buffer_kwargs.fraction_goals_are_rollout_goals': 0.2,
        'exploration_type': 'ou'
    }
)
td3_trials = get_trials(
    '/home/vitchyr/git/railrl/data/doodads3/05-12-sawyer-push-and-reach-easy/',
    criteria={
        'replay_buffer_kwargs.fraction_goals_are_rollout_goals': 1.0,
        'exploration_type': 'ou'
    }
)
my_trials = get_trials(
    '/home/vitchyr/git/railrl/data/doodads3/05-12-sawyer-push-and-reach-easy-vae-rl',
    criteria={
        'replay_kwargs.fraction_resampled_goals_are_env_goals': 0.5,
        'replay_kwargs.fraction_goals_are_rollout_goals': 0.2,
    }
)
vae_td3_trials = get_trials(
    '/home/vitchyr/git/railrl/data/doodads3/05-12-sawyer-push-and-reach-easy-vae-rl',
    criteria={
        'replay_kwargs.fraction_goals_are_rollout_goals': 1.,
    }
)


y_keys = [
    'Final  puck_distance Mean',
    'Final  hand_distance Mean',
]
plot_trials(
    {
        'State - HER TD3': state_trials,
        # 'State - TD3': td3_trials,
        'VAE - HER TD3': my_trials,
        # 'VAE - TD3': vae_td3_trials,
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
