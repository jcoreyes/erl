import argparse

import joblib

from railrl.envs.mujoco.pusher3dof import PusherEnv3DOF
from railrl.policies.base import Policy
from railrl.samplers.util import rollout
from rllab.envs.normalized_env import normalize
from rllab.misc import logger


class AveragerPolicy(Policy):
    def __init__(self, policy1, policy2):
        self.policy1 = policy1
        self.policy2 = policy2

    def get_action(self, obs):
        action1, info_dict1 = self.policy1.get_action(obs)
        action2, info_dict2 = self.policy2.get_action(obs)
        return (action1 + action2) / 2, dict(info_dict1, **info_dict2)

    def log_diagnostics(self, param):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--pause', action='store_true')
    args = parser.parse_args()

    ddpg1_snapshot_path = (
        '/home/vitchyr/git/rllab-rail/railrl/data/papers/icra2017/'
        '09-11_pusher-3dof-horizontal-2_2017_09_11_23_23_50_0039/'
        'itr_50.pkl'
    )
    ddpg2_snapshot_path = (
        '/home/vitchyr/git/rllab-rail/railrl/data/papers/icra2017/'
        '09-11_pusher-3dof-vertical-2_2017_09_11_23_24_08_0017/'
        'itr_50.pkl'
    )
    env_params = dict(
        goal=(0, -1),
    )
    env = PusherEnv3DOF(**env_params)
    env = normalize(env)
    ddpg1_snapshot_dict = joblib.load(ddpg1_snapshot_path)
    ddpg2_snapshot_dict = joblib.load(ddpg2_snapshot_path)
    policy = AveragerPolicy(
        ddpg1_snapshot_dict['policy'],
        ddpg2_snapshot_dict['policy'],
    )

    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=True,
        )
        env.log_diagnostics([path])
        policy.log_diagnostics([path])
        logger.dump_tabular()
