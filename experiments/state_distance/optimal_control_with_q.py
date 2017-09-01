"""
Choose action according to

a = argmax_{a, s'} r(s, a, s') s.t. Q(s, a, s') = 0

where r is defined specifically for the reacher env.
"""

import argparse

import joblib
import numpy as np

from railrl.policies.state_distance import SampleOptimalControlPolicy, \
    MultiStepSampleOptimalControlPolicy
from railrl.samplers.util import rollout
from railrl.torch.pytorch_util import set_gpu_mode
from rllab.misc import logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', type=int, default=10,
                        help='Number of rollouts per eval')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--plan_h', type=int, default=1,
                        help='Planning horizon')
    parser.add_argument('--discount', type=float, help='Discount Factor')
    args = parser.parse_args()

    data = joblib.load(args.file)
    print("Done loading")
    env = data['env']
    qf = data['qf']
    if args.gpu:
        set_gpu_mode(True)
        qf.cuda()
    qf.train(False)
    print("Env type:", type(env))

    if args.plan_h == 1:
        policy = SampleOptimalControlPolicy(
            qf,
            env,
            constraint_weight=100,
            sample_size=1000,
            verbose=args.verbose,
        )
    else:
        policy = MultiStepSampleOptimalControlPolicy(
            qf,
            env,
            horizon=args.plan_h,
            constraint_weight=1000,
            sample_size=100,
            verbose=args.verbose,
        )

    if 'discount' in data:
        discount = data['discount']
        if args.discount is not None:
            print("WARNING: you are overriding the saved discount factor.")
            discount = args.discount
    else:
        discount = args.discount
    policy.set_discount(discount)
    while True:
        paths = []
        for _ in range(args.num_rollouts):
            goals = env.sample_goal_states(1)
            goal = goals[0]
            if args.verbose:
                env.print_goal_state_info(goal)
            env.set_goal(goal)
            policy.set_goal(goal)
            path = rollout(
                env,
                policy,
                max_path_length=args.H,
                animated=not args.hide,
            )
            path['goal_states'] = np.repeat(
                goals,
                len(path['observations']),
                0,
            )
            paths.append(path)
        env.log_diagnostics(paths)
        logger.dump_tabular()
