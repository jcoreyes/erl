"""
Policies to be used with a state-distance Q function.
"""
import numpy as np
from torch.autograd import Variable

from railrl.torch import pytorch_util as ptu


class SamplePolicy(object):
    def __init__(self, qf, num_samples):
        self.qf = qf
        self.num_samples = num_samples

    def expand_np_to_var(self, array):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.num_samples,
            axis=0
        )
        return Variable(
            ptu.from_numpy(array_expanded).float(),
            requires_grad=False,
        )

    def get_action(self, obs, goal, discount):
        sampled_actions = np.random.uniform(-.2, .2, size=(self.num_samples, 2))
        actions = Variable(ptu.from_numpy(sampled_actions).float(), requires_grad=False)
        q_values = ptu.get_numpy(self.qf(
            self.expand_np_to_var(obs),
            actions,
            self.expand_np_to_var(goal),
            self.expand_np_to_var(np.array([discount])),
        ))
        max_i = np.argmax(q_values)
        return sampled_actions[max_i], {}

    def reset(self):
        pass


class SamplePolicyPartialOptimizer(object):
    """
    Greedy-action-partial-state implementation.

    See https://paper.dropbox.com/doc/State-Distance-QF-Results-Summary-flRwbIxt0bbUbVXVdkKzr
    for details.
    """
    def __init__(self, qf, env, num_samples):
        self.qf = qf
        self.env = env
        self.num_samples = num_samples

    def expand_np_to_var(self, array):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.num_samples,
            axis=0
        )
        return Variable(
            ptu.from_numpy(array_expanded).float(),
            requires_grad=False,
        )

    def get_action(self, obs, goal, discount):
        sampled_actions = self.env.sample_actions(self.num_samples)
        actions = ptu.np_to_var(sampled_actions)
        goals = ptu.np_to_var(
            self.env.sample_irrelevant_goal_dimensions(goal, self.num_samples)
        )

        q_values = ptu.get_numpy(self.qf(
            self.expand_np_to_var(obs),
            actions,
            goals,
            self.expand_np_to_var(np.array([discount])),
        ))
        max_i = np.argmax(q_values)
        return sampled_actions[max_i], {}

    def reset(self):
        pass


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