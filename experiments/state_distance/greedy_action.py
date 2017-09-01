import argparse

import joblib

from railrl.policies.state_distance import SamplePolicy
from railrl.algos.state_distance.state_distance_q_learning import (
    multitask_rollout,
)
from railrl.torch.pytorch_util import set_gpu_mode
from rllab.misc import logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', type=int, default=100,
                        help='Number of rollouts per eval')
    parser.add_argument('--discount', type=float,
                        help='Discount Factor')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    env = data['env']
    qf = data['qf']
    if args.gpu:
        set_gpu_mode(True)
        qf.cuda()
    qf.train(False)

    num_samples = 1000
    if args.load:
        policy = data['policy']
        policy.train(False)
    else:
        policy = SamplePolicy(qf, num_samples)

    if 'discount' in data:
        discount = data['discount']
        if args.discount is not None:
            print("WARNING: you are overriding the saved discount factor.")
            discount = args.discount
    else:
        discount = args.discount

    while True:
        paths = []
        for _ in range(args.num_rollouts):
            goal = env.sample_goal_state_for_rollout()
            if args.verbose:
                env.print_goal_state_info(goal)
            path = multitask_rollout(
                env,
                policy,
                goal,
                discount=discount,
                max_path_length=args.H,
                animated=not args.hide,
            )
            paths.append(path)
        env.log_diagnostics(paths)
        logger.dump_tabular()
