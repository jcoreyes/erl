import argparse

import joblib

from rlkit.envs.mujoco.pusher3dof import PusherEnv3DOF, get_snapshots_and_goal
from rlkit.policies.base import Policy
from rlkit.samplers.util import rollout
from rllab.envs.normalized_env import normalize
from rlkit.core import logger


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

    vertical_pos = 'middle'
    horizontal_pos = 'bottom'

    ddpg1_snapshot_path, ddpg2_snapshot_path, x_goal, y_goal = (
        get_snapshots_and_goal(
            vertical_pos=vertical_pos,
            horizontal_pos=horizontal_pos,
        )
    )
    env_params = dict(
        goal=(x_goal, y_goal),
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
