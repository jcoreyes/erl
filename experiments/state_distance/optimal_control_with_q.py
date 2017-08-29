"""
Choose action according to

a = argmax_{a, s'} r(s, a, s') s.t. Q(s, a, s') = 0

where r is defined specifically for the reacher env.
"""

import math
import argparse

import joblib
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD, Adam

import railrl.torch.pytorch_util as ptu
from railrl.samplers.util import rollout
from railrl.torch.pytorch_util import set_gpu_mode
from rllab.misc import logger


class SampleOptimalControlPolicy(object):
    """
    Do the argmax by sampling a bunch of states and acitons
    """
    def __init__(
            self,
            qf,
            env,
            constraint_weight=10,
            sample_size=100,
            verbose=False,
    ):
        self.qf = qf
        self.env = env
        self.constraint_weight = constraint_weight
        self.sample_size = sample_size
        self.verbose = verbose
        self._goal_pos_batch = None
        self._goal_batch = None
        self._discount_batch = None

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

    def set_discount(self, discount):
        self._discount_batch = self.expand_np_to_var(np.array([discount]))

    def reward(self, state, action, next_state):
        rewards_np = self.env.compute_rewards(
            ptu.get_numpy(state),
            ptu.get_numpy(action),
            ptu.get_numpy(next_state),
            ptu.get_numpy(self._goal_batch),
        )
        return ptu.np_to_var(np.expand_dims(rewards_np, 1))

    def reset(self):
        pass

    def get_action(self, obs):
        """
        Naive implementation where I just sample a bunch of a and s' and take
        the one that maximizes

            f(a, s') = r(s, a, s') - C * Q_d(s, a, s')**2

        :param obs: np.array, state/observation
        :return: np.array, action to take
        """
        sampled_actions = self.env.sample_actions(self.sample_size)
        action = ptu.np_to_var(sampled_actions)
        next_state = ptu.np_to_var(self.env.sample_states(self.sample_size))
        obs = self.expand_np_to_var(obs)
        reward = self.reward(obs, action, next_state)
        constraint_penalty = self.qf(
            obs,
            action,
            self.env.convert_obs_to_goal_states_pytorch(next_state),
            self._discount_batch,
        )**2
        score = (
            reward
            - self.constraint_weight * constraint_penalty
        )
        max_i = np.argmax(ptu.get_numpy(score))
        return sampled_actions[max_i], {}


class MultiStepSampleOptimalControlPolicy(SampleOptimalControlPolicy):
    def __init__(
            self,
            qf,
            env,
            horizon,
            **kwargs
    ):
        super().__init__(qf, env, **kwargs)
        self.horizon = horizon

    def get_action(self, obs):
        """
        Naive implementation where I just sample a bunch of a and s' and take
        the one that maximizes

            f(a, s') = \sum_{t=now}^{now+H} r(s_t, a_t, s_{t+1})
                        - C * Q_d(s_t, a_t, s_{t+1})**2

        :param obs: np.array, state/observation
        :return: np.array, action to take
        """
        state = self.expand_np_to_var(obs)
        first_sampled_actions = self.env.sample_actions(self.sample_size)
        action = ptu.np_to_var(first_sampled_actions)
        next_state = ptu.np_to_var(self.env.sample_states(self.sample_size))

        scores = []
        for i in range(self.horizon):
            reward = self.reward(state, action, next_state)
            constraint_penalty = self.qf(
                state,
                action,
                self.env.convert_obs_to_goal_states_pytorch(next_state),
                self._discount_batch,
            )**2
            score = (
                reward
                - self.constraint_weight * constraint_penalty
            )
            scores.append(score)

            action = ptu.np_to_var(
                self.env.sample_actions(self.sample_size)
            )
            state = next_state
            next_state = ptu.np_to_var(self.env.sample_states(self.sample_size))
        final_score = sum(scores)
        max_i = np.argmax(ptu.get_numpy(final_score))
        return first_sampled_actions[max_i], {}


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
