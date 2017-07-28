import argparse
import math

import joblib
import numpy as np

from railrl.algos.state_distance.state_distance_q_learning import \
    rollout_with_goal
from railrl.envs.multitask.reacher_env import FullStateVaryingWeightReacherEnv
from railrl.torch.pytorch_util import set_gpu_mode
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
        set_gpu_mode(True)
        qf.cuda()
    qf.train(False)

    num_samples = 1000
    resolution = 10
    policy = data['policy']
    policy.train(False)

    for _ in range(args.num_rollouts):
        paths = []
        for _ in range(5):
            goal = env.sample_goal_states(1)[0]
            if isinstance(env, FullStateVaryingWeightReacherEnv):
                goal[:6] = np.array([1, 1, 1, 1, 0, 0])
            env.print_goal_state_info(goal)
            env.set_goal(goal)
            paths.append(rollout_with_goal(
                env,
                policy,
                goal,
                max_path_length=args.H,
                animated=not args.hide,
            ))
        env.log_diagnostics(paths)
        logger.dump_tabular()
