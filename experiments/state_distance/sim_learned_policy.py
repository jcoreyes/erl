import argparse
import math

import joblib
import numpy as np

from railrl.algos.state_distance.state_distance_q_learning import \
    rollout_with_goal
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
    if args.load:
        policy = data['policy']
        policy.train(False)
        import ipdb; ipdb.set_trace()
    else:
        if args.grid:
            policy = GridPolicy(qf, resolution)
        else:
            policy = SamplePolicy(qf, num_samples)

    for _ in range(args.num_rollouts):
        paths = []
        for _ in range(5):
            goal = env.sample_goal_states(1)[0]
            c1 = goal[0:1]
            c2 = goal[1:2]
            s1 = goal[2:3]
            s2 = goal[3:4]
            print("Goal = ", goal)
            print("angle 1 (degrees) = ", np.arctan2(c1, s1) / math.pi * 180)
            print("angle 2 (degrees) = ", np.arctan2(c2, s2) / math.pi * 180)
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
