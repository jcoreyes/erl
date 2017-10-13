import argparse

import joblib

import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.state_distance_q_learning import (
    multitask_rollout,
)
from rllab.misc import logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--nrolls', type=int, default=1,
                        help='Number of rollout per eval')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--discount', type=float,
                        help='Discount Factor')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    env = data['env']
    num_samples = 1000
    resolution = 10
    policy = data['policy']
    policy.train(False)

    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.cuda()

    if 'discount' in data:
        discount = data['discount']
        if args.discount is not None:
            print("WARNING: you are overriding the saved discount factor.")
            discount = args.discount
    else:
        if args.discount is None:
            print("Default discount to 0.")
            discount = 0.
        else:
            discount = args.discount

    while True:
        paths = []
        for _ in range(args.nrolls):
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