"""
Choose action according to

a = argmin_a ||f(s, a) - GOAL||^2

where f is a learned forward dynamics model.
"""

import math
import argparse

import joblib
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD, Adam

import railrl.torch.pytorch_util as ptu
from railrl.envs.multitask.reacher_env import XyMultitaskSimpleStateReacherEnv, \
    GoalStateSimpleStateReacherEnv
from railrl.pythonplusplus import line_logger
from railrl.samplers.util import rollout
from railrl.torch.pytorch_util import set_gpu_mode
from rllab.misc import logger


class GreedyModelBasedPolicy(object):
    """
    Do the argmax by sampling a bunch of acitons
    """
    def __init__(
            self,
            model,
            env,
            sample_size=100,
    ):
        self.model = model
        self.env = env
        self.sample_size = sample_size
        self._goal_batch = None

    def expand_np_to_var(self, array):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.sample_size,
            axis=0
        )
        return Variable(
            ptu.from_numpy(array_expanded).float(),
            requires_grad=False,
        )

    def set_goal(self, goal):
        self._goal_batch = self.expand_np_to_var(goal)

    def reset(self):
        pass

    def get_action(self, obs):
        sampled_actions = self.env.sample_actions(self.sample_size)
        action = ptu.np_to_var(sampled_actions)
        obs = self.expand_np_to_var(obs)
        next_state_predicted = self.model(
            obs,
            action,
        )
        errors = (next_state_predicted - self._goal_batch)**2
        mean_errors = errors.mean(dim=1)
        min_i = np.argmin(ptu.get_numpy(mean_errors))
        return sampled_actions[min_i], {}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', type=int, default=100,
                        help='Number of rollouts per eval')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    print("Done loading")
    env = data['env']
    model = data['model']
    model.train(False)
    print("Env type:", type(env))

    policy = GreedyModelBasedPolicy(
        model,
        env,
        sample_size=1000,
    )
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
            paths.append(path)
        env.log_diagnostics(paths)
        logger.dump_tabular()
