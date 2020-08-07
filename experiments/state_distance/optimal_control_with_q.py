"""
Choose action according to

a = argmax_{a, s'} r(s, a, s') s.t. Q(s, a, s') = 0

where r is defined specifically for the reacher env.
"""

import argparse

import joblib
import numpy as np

from rlkit.state_distance.policies import (
    SoftOcOneStepRewardPolicy,
    TerminalRewardSampleOCPolicy,
    ArgmaxQFPolicy,
    PseudoModelBasedPolicy,
    SamplePolicyPartialOptimizer)
from rlkit.samplers.util import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.core import logger

def experiment(variant):
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=500,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', type=int, default=1,
                        help='Number of rollouts per eval')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--argmax', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--planh', type=int, default=1,
                        help='Planning horizon')
    parser.add_argument('--discount', type=float, help='Discount Factor')
    parser.add_argument('--weight', type=float, default=1.,
                        help='Constraint penalty weight')
    parser.add_argument('--nsamples', type=int, default=100,
                        help='Number of samples for optimization')
    parser.add_argument('--ngrad', type=int, default=0,
                        help='Number of gradient steps for respective policy.')
    parser.add_argument('--mb', action='store_true',
                        help='Use (pseudo-)model-based policy')
    parser.add_argument('--partial', action='store_true',
                        help='Use partial state optimizer')
    parser.add_argument('--grid', action='store_true',
                        help='Sample actions from a grid')
    parser.add_argument('--dt', help='decrement tau', action='store_true')
    parser.add_argument('--cycle', help='cycle tau', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    print("Done loading")
    env = data['env']
    qf = data['qf']
    if args.gpu:
        set_gpu_mode(True)
        qf.to(ptu.device)
    qf.train(False)
    print("Env type:", type(env))

    if args.argmax:
        policy = ArgmaxQFPolicy(
            qf,
            env,
            sample_size=args.nsamples,
            num_gradient_steps=args.ngrad,
            sample_actions_from_grid=args.grid,
        )
    elif args.mb:
        policy = PseudoModelBasedPolicy(
            qf,
            env,
            sample_size=args.nsamples,
        )
    elif args.partial:
        policy = SamplePolicyPartialOptimizer(
            qf,
            env,
            data['policy'],
            sample_size=args.nsamples,
        )
    elif args.planh == 1:
        policy = SoftOcOneStepRewardPolicy(
            qf,
            env,
            data['policy'],
            constraint_weight=args.weight,
            sample_size=args.nsamples,
            verbose=args.verbose,
            sample_actions_from_grid=args.grid,
        )
    else:
        policy = TerminalRewardSampleOCPolicy(
            qf,
            env,
            horizon=args.planh,
            constraint_weight=args.weight,
            sample_size=args.nsamples,
            verbose=args.verbose,
        )

    discount = 0
    if args.discount is not None:
        print("WARNING: you are overriding the discount factor. Right now "
              "only discount = 0 really makes sense.")
        discount = args.discount
    init_tau = discount
    while True:
        paths = []
        tau = init_tau
        policy.set_tau(tau)
        for _ in range(args.num_rollouts):
            goal = env.sample_goal_for_rollout()
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
            tau -= 1
            if tau < 0:
                if args.cycle:
                    tau = init_tau
                else:
                    tau = 0
            policy.set_tau(tau)
        env.log_diagnostics(paths)
        logger.dump_tabular()
