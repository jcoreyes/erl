import argparse

import joblib

import numpy as np
from rlkit.state_distance.policies import SamplePolicyPartialOptimizer
from rlkit.samplers.util import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.core import logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', type=int, default=5,
                        help='Number of rollouts per eval')
    parser.add_argument('--discount', type=float,
                        help='Discount Factor')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    env = data['env']
    print("Environment Type = ", type(env))
    qf = data['qf']
    if args.gpu:
        set_gpu_mode(True)
        qf.to(ptu.device)
    qf.train(False)

    if 'discount' in data:
        discount = data['discount']
        if args.discount is not None:
            print("WARNING: you are overriding the saved discount factor.")
            discount = args.discount
    else:
        discount = args.discount

    num_samples = 1000
    policy = SamplePolicyPartialOptimizer(qf, env, num_samples)

    policy.set_tau(discount)
    while True:
        paths = []
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
        env.log_diagnostics(paths)
        logger.dump_tabular()
