"""
Policies to be used with a state-distance Q function.
"""
import abc
import numpy as np
from itertools import product
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim

from scipy import optimize

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
    def __init__(self, sample_size, env, sample_actions_from_grid=False):
        super().__init__()
        self.sample_size = sample_size
        self.env = env
        self.sample_actions_from_grid = sample_actions_from_grid
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
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

    def sample_actions(self):
        if self.sample_actions_from_grid:
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

    def sample_states(self):
        return self.env.sample_states(self.sample_size)


class SamplePolicyPartialOptimizer(SampleBasedUniversalPolicy, nn.Module):
    """
    Greedy-action-partial-state implementation.

    Make it sublcass nn.Module so that calls to `train` and `cuda` get
    propagated to the sub-networks

    See https://paper.dropbox.com/doc/State-Distance-QF-Results-Summary-flRwbIxt0bbUbVXVdkKzr
    for details.
    """
    def __init__(self, qf, env, sample_size=100, **kwargs):
        nn.Module.__init__(self)
        super().__init__(sample_size, env, **kwargs)
        self.qf = qf

    def get_action(self, obs):
        sampled_actions = self.sample_actions()
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
            constraint_weight=1,
            sample_size=100,
            verbose=False,
            **kwargs
    ):
        nn.Module.__init__(self)
        super().__init__(sample_size, env, **kwargs)
        self.qf = qf
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
        sampled_actions = self.sample_actions()
        action = ptu.np_to_var(sampled_actions)
        next_state = ptu.np_to_var(self.sample_states())
        obs = self.expand_np_to_var(obs)
        reward = self.reward(obs, action, next_state)
        constraint_penalty = self.qf(
            obs,
            action,
            self.env.convert_obs_to_goal_states_pytorch(next_state),
            self._discount_batch,
        )
        score = (
            reward
            + self.constraint_weight * constraint_penalty
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
        first_sampled_actions = self.sample_actions()
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
            **kwargs
    ):
        nn.Module.__init__(self)
        super().__init__(sample_size, env, **kwargs)
        self.qf = qf
        self.learning_rate = learning_rate
        self.num_gradient_steps = num_gradient_steps

    def get_action(self, obs):
        action_inits = self.sample_actions()
        actions = ptu.np_to_var(action_inits, requires_grad=True)
        obs = self.expand_np_to_var(obs)
        optimizer = optim.Adam([actions], self.learning_rate)
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


class PseudoModelBasedPolicy(SampleBasedUniversalPolicy, nn.Module):
    """
    1. Sample actions
    2. Optimize over next state (according to a Q function)
    3. Compare next state with desired next state to choose action
    """
    def __init__(
            self,
            qf,
            env,
            sample_size=100,
            learning_rate=1e-1,
            num_gradient_steps=100,
            state_optimizer='adam',
            **kwargs
    ):
        nn.Module.__init__(self)
        super().__init__(sample_size, env, **kwargs)
        self.qf = qf
        self.learning_rate = learning_rate
        self.num_optimization_steps = num_gradient_steps
        self.state_optimizer = state_optimizer
        self.observation_dim = self.env.observation_space.low.size

    def get_next_states_np(self, states, actions):
        if self.state_optimizer == 'adam':
            next_states_np = np.zeros((self.sample_size, self.observation_dim))
            next_states = ptu.np_to_var(next_states_np, requires_grad=True)
            optimizer = optim.Adam([next_states], self.learning_rate)

            for _ in range(self.num_optimization_steps):
                losses = -self.qf(
                    states,
                    actions,
                    next_states,
                    self._discount_batch,
                )
                loss = losses.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return ptu.get_numpy(next_states)
        elif self.state_optimizer == 'lbfgs':
            next_states = []
            for i in range(len(states)):
                state = states[i:i+1, :]
                action = actions[i:i+1, :]
                loss_f = self.create_loss(state, action, return_gradient=True)
                results = optimize.fmin_l_bfgs_b(
                    loss_f,
                    np.zeros((1, self.observation_dim)),
                    maxiter=self.num_optimization_steps,
                )
                next_state = results[0]
                next_states.append(next_state)
            next_states = np.array(next_states)
            return next_states
        elif self.state_optimizer == 'fmin':
            next_states = []
            for i in range(len(states)):
                state = states[i:i+1, :]
                action = actions[i:i+1, :]
                loss_f = self.create_loss(state, action)
                results = optimize.fmin(
                    loss_f,
                    np.zeros((1, self.observation_dim)),
                    maxiter=self.num_optimization_steps,
                )
                next_state = results[0]
                next_states.append(next_state)
            next_states = np.array(next_states)
            return next_states
        else:
            raise Exception(
                "Unknown state optimizer mode: {}".format(self.state_optimizer)
            )

    def create_loss(self, state, action, return_gradient=False):
        def f(next_state_np):
            next_state = ptu.np_to_var(
                np.expand_dims(next_state_np, 0),
                requires_grad=True,
            )
            loss = - self.qf(
                state,
                action,
                next_state,
                self._discount_expanded_torch
            )
            loss.backward()
            loss_np = ptu.get_numpy(loss)
            gradient_np = ptu.get_numpy(next_state.grad)
            if return_gradient:
                return loss_np, gradient_np.astype('double')
            else:
                return loss_np
        return f

    def get_action(self, obs):
        sampled_actions = self.sample_actions()
        states = self.expand_np_to_var(obs)
        actions = ptu.np_to_var(sampled_actions)
        next_states = self.get_next_states_np(states, actions)

        distances = np.sum(
            (next_states - self._goal_np)**2,
            axis=1
        )
        best_action = np.argmin(distances)
        return sampled_actions[best_action, :], {}


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


def scale_input_output(function):
    def new_function(self, normalized_x):
        actual_output = function(actual_x)
        normalized_output = (
            (actual_output - self.lower_bounds) * 2 /
            (self.upper_bounds - self.lower_bounds)
            - 1
        )
        return normalized_output
    return new_function


def scale_inputs_outputs(function):
    def new_function(self, normalized_x, optional=None):
        actual_x = (
            self.lower_bounds + (normalized_x + 1) * 0.5 * (
                self.upper_bounds - self.lower_bounds
            )
        )
        actual_output = function(actual_x)
        normalized_output = (
            (actual_output - self.lower_bounds) * 2 /
            (self.upper_bounds - self.lower_bounds)
            - 1
        )
        return normalized_output
    return new_function


class ConstrainedOptimizationOCPolicy(UniversalPolicy, nn.Module):
    """
    Solve

    \pi(s, g) = argmin_a min_{s'} ||s' - g||^2
        subject to Q(s, a, s') =0

    with sequential quadratic programming.
    """
    def __init__(
            self,
            qf,
            env,
            solver_params=None,
    ):
        nn.Module.__init__(self)
        super().__init__()
        self.qf = qf
        self.env = env
        self.solver_params = solver_params
        self.action_dim = self.env.action_space.low.size
        self.observation_dim = self.env.observation_space.low.size
        self.last_solution = np.zeros(self.action_dim + self.observation_dim)
        self.lower_bounds = np.hstack((
            self.env.action_space.low,
            self.env.observation_space.low,
        ))
        self.upper_bounds = np.hstack((
                self.env.action_space.high,
                self.env.observation_space.high,
        ))
        self.lb_obs = self.env.observation_space.low
        self.ub_obs = self.env.observation_space.high
        self.bounds = (-1, 1)
        import ipdb; ipdb.set_trace()
        self.bounds = list(zip(self.lower_bounds, self.upper_bounds))
        self.constraints = {
            'type': 'eq',
            'fun': self.constraint_fctn,
            'jac': self.constraint_jacobian,
        }
        # for i, lb in enumerate(self.lower_bounds):
        #     constraints.append({
        #         'type': 'ineq',
        #         'fun': lambda x: x[i] - lb.copy(),
        #         'jac': lambda x: 1,
        #     })
        # for i, ub in enumerate(self.upper_bounds):
        #     constraints.append({
        #         'type': 'ineq',
        #         'fun': lambda x: ub.copy() - x,
        #         'jac': lambda x: -1,
        #     })
        # import ipdb; ipdb.set_trace()

        self.actual_to_normalized_gradient = (
            2 / (self.upper_bounds - self.lower_bounds)
        )

    def normalized_to_actual(self, normalized):
        if len(normalized) == self.observation_dim + self.action_dim:
            return (
                self.lower_bounds + (normalized + 1) * 0.5 * (
                    self.upper_bounds - self.lower_bounds
                )
            )
        elif len(normalized) == self.observation_dim:
            return (
                self.lb_obs + (normalized + 1) * 0.5 * (
                    self.ub_obs - self.lb_obs
                )
            )
        else:
            raise Exception(
                "Do not know how to un-normalize array of shape {}".format(
                    normalized.shape
                )
            )

    def actual_to_normalized(self, actual):
        return (
            (actual - self.lower_bounds) * 2 /
            (self.upper_bounds - self.lower_bounds)
            - 1
        )

    def cost_function(self, action_and_next_state_flat):
        # action_and_next_state_flat = self.normalized_to_actual(
        #     action_and_next_state_flat
        # )
        next_state = action_and_next_state_flat[self.action_dim:]
        return np.linalg.norm(next_state - self._goal_np)

    def constraint_fctn(self, action_next_state_flat, state=None):
        # action_next_state_flat = self.normalized_to_actual(
        #     action_next_state_flat
        # )
        # state = self.normalized_to_actual(state)

        state = ptu.np_to_var(state)
        action_next_state_flat = ptu.np_to_var(
            action_next_state_flat,
            requires_grad=True,
        )
        action = action_next_state_flat[0:self.action_dim]
        next_state = action_next_state_flat[self.action_dim:]

        qvalue = ptu.get_numpy(
            self.qf(
                state.unsqueeze(0),
                action.unsqueeze(0),
                next_state.unsqueeze(0),
                self._discount_expanded_torch
            ).squeeze(0)
        )
        return qvalue[0]

    def constraint_jacobian(self, action_next_state_flat, state=None):
        # action_next_state_flat = self.normalized_to_actual(
        #     action_next_state_flat
        # )
        # state = self.normalized_to_actual(state)

        state = ptu.np_to_var(state)
        action_next_state_flat = ptu.np_to_var(
            action_next_state_flat,
            requires_grad=True,
        )
        action = action_next_state_flat[0:self.action_dim]
        next_state = action_next_state_flat[self.action_dim:]

        q_value = self.qf(
            state.unsqueeze(0),
            action.unsqueeze(0),
            next_state.unsqueeze(0),
            self._discount_expanded_torch
        )
        q_value.backward()
        return (
            ptu.get_numpy(action_next_state_flat.grad)
            # * self.actual_to_normalized_gradient
        )

    def reset(self):
        self.last_solution = np.zeros(self.action_dim + self.observation_dim)

    def get_action(self, obs):
        self.constraints['args'] = (obs, )
        result = optimize.minimize(
            self.cost_function,
            self.last_solution,
            constraints=self.constraints,
            method='SLSQP',
            options=self.solver_params,
            bounds=self.bounds,
        )
        # action = self.normalize_to_actual(result.x)[:self.action_dim]
        action = result.x[:self.action_dim]
        if np.isnan(action).any():
            import ipdb; ipdb.set_trace()
        self.last_solution = result.x
        return action, {}
