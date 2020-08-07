import argparse
import matplotlib.pyplot as plt
import numpy as np

import joblib

from rlkit.policies.simple import RandomPolicy
from rlkit.state_distance.rollout_util import multitask_rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--pause', action='store_true')
    args = parser.parse_args()
    if args.pause:
        import ipdb; ipdb.set_trace()

    data = joblib.load(args.file)
    env = data['env']
    qf = data['qf']
    policy = data['policy']
    tdm_policy = data['trained_policy']
    random_policy = RandomPolicy(env.action_space)
    vf = data['vf']
    path = multitask_rollout(
        env,
        # tdm_policy,
        random_policy,
        init_tau=0,
        max_path_length=100,
        animated=True,
    )
    obs = path['observations']
    next_obs = path['next_observations']
    num_steps_left = np.zeros((obs.shape[0], 1))

    actions = tdm_policy.eval_np(obs, next_obs, num_steps_left)
    qvals = qf.eval_np(obs, actions, next_obs, num_steps_left)
    vvals = vf.eval_np(obs, next_obs, num_steps_left)

    num_dims = qvals.shape[1]
    for dim in range(num_dims):
        plt.subplot(num_dims, 1, dim + 1)
        plt.plot(qvals[:, dim], label='q')
        plt.plot(vvals[:, dim], label='v')
        plt.plot(qvals[:, dim] - vvals[:, dim], label='q - v')
        plt.title("Dim %d" % dim)
    plt.legend()
    plt.show()
