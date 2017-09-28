"""
Use a learned dynamics model to solve a task.
"""
import argparse
import numpy as np

import joblib

from railrl.policies.model_based import MultistepModelBasedPolicy
from railrl.samplers.util import rollout
from rllab.misc import logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', type=int, default=1,
                        help='Number of rollouts per eval')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    print("Done loading")
    env = data['env']
    model = data['model']
    model.train(False)
    print("Env:", env)

    policy = MultistepModelBasedPolicy(
        model,
        env,
        sample_size=1000,
        planning_horizon=5,
        model_learns_deltas=True,
    )
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
                axis=0,
            )
            paths.append(path)
        env.log_diagnostics(paths)
        logger.dump_tabular()