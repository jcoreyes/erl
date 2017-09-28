"""
Choose action according to

a = argmax_{a, s'} r(s, a, s') s.t. Q(s, a, s') = 0

where r is defined specifically for the reacher env.
"""

import argparse

import joblib
import numpy as np

from railrl.policies.state_distance import SampleOptimalControlPolicy, \
    TerminalRewardSampleOCPolicy, ArgmaxQFPolicy, BeamSearchMultistepSampler
from railrl.samplers.util import rollout
from railrl.torch.pytorch_util import set_gpu_mode
from rllab.misc import logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', type=int, default=1,
                        help='Number of rollouts per eval')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--argmax', action='store_true')
    parser.add_argument('--beam', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--plan_h', type=int, default=1,
                        help='Planning horizon')
    parser.add_argument('--discount', type=float, help='Discount Factor')
    parser.add_argument('--weight', type=float, default=1000.,
                        help='Constraint penalty weight')
    parser.add_argument('--nsamples', type=int, default=1000,
                        help='Number of samples for optimization')
    parser.add_argument('--ngrad', type=int, default=100,
                        help='Number of gradient steps for respective policy.')
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

    if args.argmax:
        policy = ArgmaxQFPolicy(
            qf,
            env,
            sample_size=args.nsamples,
            num_gradient_steps=args.ngrad,
            sample_actions_from_grid=True,
        )
    elif args.beam:
        policy = BeamSearchMultistepSampler(
            qf,
            env,
            args.plan_h,
        )
    elif args.plan_h == 1:
        policy = SampleOptimalControlPolicy(
            qf,
            env,
            constraint_weight=args.weight,
            sample_size=args.nsamples,
            verbose=args.verbose,
        )
    else:
        policy = TerminalRewardSampleOCPolicy(
            qf,
            env,
            horizon=args.plan_h,
            constraint_weight=args.weight,
            sample_size=args.nsamples,
            verbose=args.verbose,
        )

    discount = 0
    if args.discount is not None:
        print("WARNING: you are overriding the discount factor. Right now "
              "only discount = 0 really makes sense.")
        discount = args.discount
    policy.set_discount(discount)
    while True:
        paths = []
        for _ in range(args.num_rollouts):
            goal = env.sample_goal_state_for_rollout()
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
                np.expand_dims(goal, 0),
                len(path['observations']),
                0,
            )
            paths.append(path)
        env.log_diagnostics(paths)
        logger.dump_tabular()
