"""
Policies to be used with a state-distance Q function.
"""
import abc
import numpy as np
from torch.autograd import Variable

from railrl.policies.base import ExplorationPolicy
from railrl.torch import pytorch_util as ptu


class SampleBasedUniversalPolicy(ExplorationPolicy, metaclass=abc.ABCMeta):
    def __init__(self, sample_size):
        self.sample_size = sample_size
        self._goal_np = None
        self._goal_batch = None
        self._discount_np = None
        self._discount_batch = None

    def set_goal(self, goal):
        self._goal_np = goal
        self._goal_batch = self.expand_np_to_var(goal)

    def set_discount(self, discount):
        self._discount_np = discount
        self._discount_batch = self.expand_np_to_var(np.array([discount]))

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


class SamplePolicyPartialOptimizer(SampleBasedUniversalPolicy):
    """
    Greedy-action-partial-state implementation.

    See https://paper.dropbox.com/doc/State-Distance-QF-Results-Summary-flRwbIxt0bbUbVXVdkKzr
    for details.
    """
    def __init__(self, qf, env, sample_size=100):
        super().__init__(sample_size)
        self.qf = qf
        self.env = env

    def get_action(self, obs):
        sampled_actions = self.env.sample_actions(self.sample_size)
        actions = ptu.np_to_var(sampled_actions)
        goals = ptu.np_to_var(
            self.env.sample_irrelevant_goal_dimensions(
                self._goal_np, self.sample_size
            )
        )

        q_values = ptu.get_numpy(self.qf(
            self.expand_np_to_var(obs),
            actions,
            goals,
            self.expand_np_to_var(np.array([self._discount_np])),
        ))
        max_i = np.argmax(q_values)
        return sampled_actions[max_i], {}


class SampleOptimalControlPolicy(SampleBasedUniversalPolicy):
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
        super().__init__(sample_size)
        self.qf = qf
        self.env = env
        self.constraint_weight = constraint_weight
        self.sample_size = sample_size
        self.verbose = verbose

    def reward(self, state, action, next_state):
        rewards_np = self.env.compute_rewards(
            ptu.get_numpy(state),
            ptu.get_numpy(action),
            ptu.get_numpy(next_state),
            ptu.get_numpy(self._goal_batch),
        )
        return ptu.np_to_var(np.expand_dims(rewards_np, 1))

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