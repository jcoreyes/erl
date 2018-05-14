import matplotlib.pyplot as plt

from railrl.misc.data_processing import get_trials
from railrl.misc.plot_util import plot_trials

plt.style.use("ggplot")


vae_trials = get_trials(
    '/home/vitchyr/git/railrl/data/doodads3/05-12-sawyer-reach-vae-rl-log-prob-rewards-2',
    criteria={
        'replay_kwargs.fraction_goals_are_env_goals': 0.5,
        'replay_kwargs.fraction_goals_are_rollout_goals': 0.2,
        'reward_params.type': 'latent_distance'
    }
)
# vae_trials = get_trials(
#     '/home/vitchyr/git/railrl/data/doodads3/05-12-sawyer-reach-vae-rl-reproduce-2/',
#     criteria={
#         'replay_kwargs.fraction_goals_are_env_goals': 0.5,
#         'replay_kwargs.fraction_goals_are_rollout_goals': 0.2,
#     }
# )


y_keys = [
    'Final  distance Mean',
]
plot_trials(
    {
        # 'State - HER TD3': state_trials,
        # 'State - TD3': td3_trials,
        'VAE - HER TD3': vae_trials,
        # 'VAE - TD3': vae_td3_trials,
    },
    y_keys=y_keys,
    # x_key=x_key,
)

plt.xlabel('Number of Environment Steps Total')
plt.ylabel('Final distance to Goal')
plt.savefig('/home/vitchyr/git/railrl/experiments/vitchyr/nips2018/plots'
            '/reach.jpg')
plt.show()

# plt.savefig("/home/ashvin/data/s3doodad/media/plots/pusher2d.pdf")
