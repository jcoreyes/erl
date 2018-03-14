import argparse
import numpy as np
import matplotlib.pyplot as plt

import joblib

from railrl.misc.visualization_util import make_heat_map, plot_heatmap
from railrl.policies.simple import RandomPolicy
from railrl.state_distance.rollout_util import multitask_rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--pause', action='store_true')
    args = parser.parse_args()
    if args.pause:
        import ipdb; ipdb.set_trace()

    file = (
        '/home/vitchyr/git/railrl/data/local/name-of-beta-learning-experiment/name-of-beta-learning-experiment_2018_03_12_17_45_52_0000--s-0/params.pkl'
        or args.file
    )
    data = joblib.load(file)
    beta_q = data['beta_q']
    env = data['env']

    random_policy = RandomPolicy(env.action_space)
    path = multitask_rollout(
        env,
        random_policy,
        init_tau=0,
        max_path_length=2,
        # animated=True,
    )
    path_obs = path['observations']
    path_actions = path['actions']
    path_next_obs = path['next_observations']
    num_steps_left = np.zeros((1, 1))

    obs = path_obs[0:1]
    next_obs = path_next_obs[0:1]
    actions = path_actions[0:1]
    beta_values = []

    def beta_eval(a1, a2):
        actions = np.array([[a1, a2]])
        return beta_q.eval_np(
            observations=np.array([[-4, 4]]),
            actions=actions,
            goals=np.array([[-3, 4]]),
            num_steps_left=num_steps_left
        )[0, 0]

    print("obs:", obs)
    print("true action:", actions)
    print("next obs:", next_obs)
    heatmap = make_heat_map(beta_eval, [-1, 1], [-1, 1])
    plot_heatmap(heatmap)
    plt.show()

