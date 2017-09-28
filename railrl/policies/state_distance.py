"""
Policies to be used with a state-distance Q function.
"""
import abc
import numpy as np
from itertools import product
from torch import nn
from torch.autograd import Variable
from torch import optim

from railrl.policies.base import ExplorationPolicy, Policy
from railrl.torch import pytorch_util as ptu


class UniversalPolicy(Policy, metaclass=abc.ABCMeta):
    def __init__(self):
        self._goal_np = None
        self._goal_expanded_torch = None
        self._discount_np = None
        self._discount_expanded_torch = None

    def set_goal(self, goal_np):
        self._goal_np = goal_np
        self._goal_expanded_torch = ptu.np_to_var(
            np.expand_dims(goal_np, 0)
        )

    def set_discount(self, discount):
        self._discount_np = discount
        self._discount_expanded_torch = ptu.np_to_var(
            np.array([[discount]])
        )


class SampleBasedUniversalPolicy(
    UniversalPolicy, ExplorationPolicy, metaclass=abc.ABCMeta
):
    def __init__(self, sample_size):
        super().__init__()
        self.sample_size = sample_size
        self._goal_batch = None
        self._discount_batch = None

    def set_goal(self, goal_np):
        super().set_goal(goal_np)
        self._goal_batch = self.expand_np_to_var(goal_np)

    def set_discount(self, discount):
        super().set_discount(discount)
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


class SamplePolicyPartialOptimizer(SampleBasedUniversalPolicy, nn.Module):
    """
    Greedy-action-partial-state implementation.

    Make it sublcass nn.Module so that calls to `train` and `cuda` get
    propagated to the sub-networks

    See https://paper.dropbox.com/doc/State-Distance-QF-Results-Summary-flRwbIxt0bbUbVXVdkKzr
    for details.
    """
    def __init__(self, qf, env, sample_size=100):
        nn.Module.__init__(self)
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


class SampleOptimalControlPolicy(SampleBasedUniversalPolicy, nn.Module):
    """
    Do the argmax by sampling a bunch of states and actions

    Make it sublcass nn.Module so that calls to `train` and `cuda` get
    propagated to the sub-networks
    """
    def __init__(
            self,
            qf,
            env,
            constraint_weight=10,
            sample_size=100,
            verbose=False,
    ):
        nn.Module.__init__(self)
        super().__init__(sample_size)
        self.qf = qf
        self.env = env
        self.constraint_weight = constraint_weight
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
        print("reward", reward)
        print("constraint_penalty", constraint_penalty)
        score = (
            reward
            - self.constraint_weight * constraint_penalty
        )
        max_i = np.argmax(ptu.get_numpy(score))
        return sampled_actions[max_i], {}


class TerminalRewardSampleOCPolicy(SampleOptimalControlPolicy, nn.Module):
    """
    Want to implement:

        a = \argmax_{a_T} \max_{a_{1:T-1}, s_{1:T+1}} r(s_{T+1})

        s.t.  Q(s_t, a_t, s_g=s_{t+1}, tau=0) = 0, t=1, T

    Softened version of this:

        a = \argmax_{a_T} \max_{a_{1:T-1}, s_{1:T+1}} r(s_{T+1})
         - C * \sum_{t=1}^T Q(s_t, a_t, s_g=s_{t+1}, tau=0)^2

          = \argmax_{a_T} \max_{a_{1:T-1}, s_{1:T+1}} f(a_{1:T}, s_{1:T+1})

    Naive implementation where I just sample a bunch of a's and s's and take
    the max of this function f.

    Make it sublcass nn.Module so that calls to `train` and `cuda` get
    propagated to the sub-networks

    :param obs: np.array, state/observation
    :return: np.array, action to take
    """
    def __init__(
            self,
            qf,
            env,
            horizon,
            **kwargs
    ):
        nn.Module.__init__(self)
        super().__init__(qf, env, **kwargs)
        self.horizon = horizon
        self._tau_batch = self.expand_np_to_var(np.array([0]))

    def get_action(self, obs):
        state = self.expand_np_to_var(obs)
        first_sampled_actions = self.env.sample_actions(self.sample_size)
        action = ptu.np_to_var(first_sampled_actions)
        next_state = ptu.np_to_var(self.env.sample_states(self.sample_size))

        penalties = []
        for i in range(self.horizon):
            constraint_penalty = self.qf(
                state,
                action,
                self.env.convert_obs_to_goal_states_pytorch(next_state),
                self._tau_batch,
            )**2
            penalties.append(
                - self.constraint_weight * constraint_penalty
            )

            action = ptu.np_to_var(
                self.env.sample_actions(self.sample_size)
            )
            state = next_state
            next_state = ptu.np_to_var(self.env.sample_states(self.sample_size))
        reward = self.reward(state, action, next_state)
        final_score = reward + sum(penalties)
        max_i = np.argmax(ptu.get_numpy(final_score))
        return first_sampled_actions[max_i], {}


class ArgmaxQFPolicy(SampleBasedUniversalPolicy, nn.Module):
    """
    pi(s, g) = \argmax_a Q(s, a, g)

    Implemented by initializing a bunch of actions and doing gradient descent on
    them.

    This should be the same as a policy learned in DDPG.
    This is basically a sanity check.
    """
    def __init__(
            self,
            qf,
            env,
            sample_size=100,
            learning_rate=1e-1,
            num_gradient_steps=10,
            sample_actions_on_grid=False,
    ):
        nn.Module.__init__(self)
        super().__init__(sample_size)
        self.qf = qf
        self.env = env
        self.learning_rate = learning_rate
        self.num_gradient_steps = num_gradient_steps
        self.sample_actions_on_grid = sample_actions_on_grid
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

    def get_action(self, obs):
        sampled_actions = self.sample_actions()
        actions = ptu.np_to_var(sampled_actions, requires_grad=True)
        obs = self.expand_np_to_var(obs)
        optimizer = optim.SGD([actions], self.learning_rate)
        losses = -self.qf(
            obs,
            actions,
            self._goal_batch,
            self._discount_batch,
        )
        for _ in range(self.num_gradient_steps):
            loss = losses.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses = -self.qf(
                obs,
                actions,
                self._goal_batch,
                self._discount_batch,
            )
        losses_np = ptu.get_numpy(losses)
        best_action_i = np.argmin(losses_np)
        return ptu.get_numpy(actions[best_action_i, :]), {}

    def sample_actions(self):
        if self.sample_actions_on_grid:
            action_dim = self.env.action_dim
            resolution = int(np.power(self.sample_size, 1./action_dim))
            values = []
            for dim in range(action_dim):
                values.append(np.linspace(
                    self.action_low[dim],
                    self.action_high[dim],
                    num=resolution
                ))
            actions = np.array(list(product(*values)))
            if len(actions) < self.sample_size:
                # Add extra actions in case the grid can't perfectly create
                # `self.sample_size` actions. e.g. sample_size is 30, but the
                # grid is 5x5.
                actions = np.concatenate(
                    (
                        actions,
                        self.env.sample_actions(
                            self.sample_size - len(actions)
                        ),
                    ),
                    axis=0,
                )
            return actions
        else:
            return self.env.sample_actions(self.sample_size)


class BeamSearchMultistepSampler(SampleBasedUniversalPolicy, nn.Module):
    def __init__(
            self,
            qf,
            env,
            horizon,
            num_particles=10,
            sample_size=100,
    ):
        nn.Module.__init__(self)
        super().__init__(sample_size)
        self.qf = qf
        self.env = env
        self.horizon = horizon
        self.num_particles = num_particles

    def get_action(self, obs):
        obs = self.expand_np_to_var(obs)
        first_sampled_actions = self.env.sample_actions(self.sample_size)
        action = ptu.np_to_var(first_sampled_actions)
        next_state = ptu.np_to_var(self.env.sample_states(self.sample_size))

        penalties = []
        for i in range(self.horizon):
            q_values = self.qf(
                state,
                action,
                self._goal_batch,
                self._discount_batch,
            )
            penalties.append(
                - self.constraint_weight * constraint_penalty
            )

            action = ptu.np_to_var(
                self.env.sample_actions(self.sample_size)
            )
            state = next_state
            next_state = ptu.np_to_var(self.env.sample_states(self.sample_size))
