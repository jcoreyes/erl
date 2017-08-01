import argparse
import math

import joblib
import numpy as np

from railrl.algos.state_distance.state_distance_q_learning import (
    rollout_with_goal,
    rollout,
)
from railrl.envs.multitask.reacher_env import FullStateVaryingWeightReacherEnv
import railrl.torch.pytorch_util as ptu
from rllab.misc import logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', type=int, default=100,
                        help='Total number of rollout')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    env = data['env']
    qf = data['qf']
    if args.gpu:
        ptu.set_gpu_mode(True)
        qf.cuda()
    qf.train(False)

    num_samples = 1000
    resolution = 10
    policy = data['policy']
    policy.train(False)

    for _ in range(args.num_rollouts):
        paths = []
        for _ in range(5):
            goals = env.sample_goal_states(1)
            goal = goals[0]
            if isinstance(env, FullStateVaryingWeightReacherEnv):
                goal[:6] = np.array([1, 1, 1, 1, 0, 0])
            env.print_goal_state_info(goal)
            env.set_goal(goal)
            path = rollout(
                env,
                policy,
                goal,
                discount=0,
                max_path_length=args.H,
                animated=not args.hide,
            )
            path['goal_states'] = goals.repeat(len(path['observations']), 0)
            paths.append(path)
        env.log_diagnostics(paths)
        logger.dump_tabular()
