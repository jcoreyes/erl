import numpy as np
import torch
from torch import nn
from scipy import optimize

from railrl.networks.state_distance import StructuredUniversalQfunction
from railrl.policies.state_distance import (
    UniversalPolicy,
    SampleBasedUniversalPolicy,
)
from railrl.torch import pytorch_util as ptu
from rllab.misc import logger


class MultistepModelBasedPolicy(SampleBasedUniversalPolicy, nn.Module):
    """
    Choose action according to

    a = argmin_{a_0} argmin_{a_{0:H-1}}||s_H - GOAL||^2

    where

        s_{i+1} = f(s_i, a_i)

    for i = 1, ..., H-1 and f is a learned forward dynamics model. In other
    words, to a multi-step optimization.

    Approximate the argmin by sampling a bunch of actions
    """
    def __init__(
            self,
            model,
            env,
            sample_size=100,
            action_penalty=0,
            planning_horizon=1,
            model_learns_deltas=True,
    ):
        super().__init__(sample_size, env)
        nn.Module.__init__(self)
        self.model = model
        self.env = env
        self.action_penalty = action_penalty
        self.planning_horizon = planning_horizon
        self.model_learned_deltas = model_learns_deltas

    def get_action(self, obs):
        sampled_actions = self.env.sample_actions(self.sample_size)
        first_sampled_action = sampled_actions
        action = ptu.np_to_var(sampled_actions)
        obs = self.expand_np_to_var(obs)
        obs_predicted = obs
        for i in range(self.planning_horizon):
            if i > 0:
                sampled_actions = self.env.sample_actions(self.sample_size)
                action = ptu.np_to_var(sampled_actions)
            if self.model_learned_deltas:
                obs_delta_predicted = self.model(
                    obs_predicted,
                    action,
                )
                obs_predicted += obs_delta_predicted
            else:
                obs_predicted = self.model(
                    obs_predicted,
                    action,
                )
        next_goal_state_predicted = (
            self.env.convert_obs_to_goal_states_pytorch(
                obs_predicted
            )
        )
        errors = (next_goal_state_predicted - self._goal_batch)**2
        mean_errors = ptu.get_numpy(errors.mean(dim=1))
        score = mean_errors + self.action_penalty * np.linalg.norm(
            sampled_actions,
            axis=1
        )
        min_i = np.argmin(score)
        return first_sampled_action[min_i], {}


class SQPModelBasedPolicy(UniversalPolicy, nn.Module):
    def __init__(
            self,
            model,
            env,
            model_learns_deltas=True,
            solver_params=None,
            model_is_structured_qf=False,
            planning_horizon=1,
    ):
        super().__init__()
        nn.Module.__init__(self)
        self.model = model
        self.env = env
        self.model_learns_deltas = model_learns_deltas
        self.solver_params = solver_params
        self.planning_horizon = planning_horizon
        self.model_is_structured_qf = model_is_structured_qf

        self.action_dim = self.env.action_space.low.size
        self.observation_dim = self.env.observation_space.low.size
        self.last_solution = None
        self.lower_bounds = np.hstack((
            np.tile(self.env.action_space.low, self.planning_horizon),
            np.tile(self.env.observation_space.low, self.planning_horizon),
        ))
        self.upper_bounds = np.hstack((
            np.tile(self.env.action_space.high, self.planning_horizon),
            np.tile(self.env.observation_space.high, self.planning_horizon),
        ))
        self.bounds = list(zip(self.lower_bounds, self.upper_bounds))
        self.constraints = {
            'type': 'eq',
            'fun': self.constraint_fctn,
            'jac': self.constraint_jacobian,
        }

    def split(self, x):
        """
        :param x: vector passed to optimization
        :return: tuple
            - actions np.array, shape [planning_horizon X action_dim]
            - next_states np.array, shape [planning_horizon X obs_dim]
        """
        all_actions = x[:self.action_dim * self.planning_horizon]
        all_next_states = x[self.action_dim * self.planning_horizon:]
        if isinstance(x, np.ndarray):
            return (
                all_actions.reshape(self.planning_horizon, self.action_dim),
                all_next_states.reshape(self.planning_horizon, self.observation_dim)
            )
        else:
            return (
                all_actions.view(self.planning_horizon, self.action_dim),
                all_next_states.view(self.planning_horizon, self.observation_dim)
            )

    def cost_function(self, x):
        _, all_next_states = self.split(x)
        last_state = all_next_states[-1, :]
        return np.sum((last_state - self._goal_np)**2)

    def cost_jacobian(self, x):
        jacobian = np.zeros_like(x)
        _, all_next_states = self.split(x)
        last_state = all_next_states[-1, :]
        # Assuming the last `self.observation_dim` part of x is the last state
        jacobian[-self.observation_dim:] = (
            2 * (last_state - self._goal_np)
        )
        return jacobian

    def constraint_fctn(self, x, state=None):
        state = ptu.np_to_var(state)
        x = ptu.np_to_var(x, requires_grad=False)
        all_actions, all_next_states = self.split(x)

        loss = 0
        state_predicted = state.unsqueeze(0)
        for i in range(self.planning_horizon):
            action = all_actions[i:i+1, :]
            next_state_predicted = self.get_next_state_predicted(
                state_predicted,
                action,
            )
            next_state = all_next_states[i:i+1, :]
            loss += torch.norm(next_state - next_state_predicted, p=2)
            state_predicted = next_state_predicted
        return ptu.get_numpy(loss)

    def constraint_jacobian(self, x, state=None):
        state = ptu.np_to_var(state)
        x = ptu.np_to_var(x, requires_grad=True)
        all_actions, all_next_states = self.split(x)

        loss = 0
        state_predicted = state.unsqueeze(0)
        for i in range(self.planning_horizon):
            action = all_actions[i:i+1, :]
            next_state_predicted = self.get_next_state_predicted(
                state_predicted,
                action,
            )
            next_state = all_next_states[i:i+1, :]
            loss += torch.norm(next_state - next_state_predicted, p=2)
            state_predicted = next_state_predicted
        loss.squeeze(0).backward()
        return ptu.get_numpy(x.grad)

    def get_next_state_predicted(self, state, action):
        if self.model_is_structured_qf:
            discount = ptu.np_to_var(
                np.array([[0]])
            )
            return self.model(state, action, None, discount, True)
        if self.model_learns_deltas:
            return state + self.model(state, action)
        else:
            return self.model(state, action)

    def reset(self):
        self.last_solution = None

    def get_action(self, obs):
        if self.last_solution is None:
            self.last_solution = np.hstack((
                np.zeros(self.action_dim * self.planning_horizon),
                np.tile(obs, self.planning_horizon),
            ))
        self.constraints['args'] = (obs, )
        result = optimize.minimize(
            self.cost_function,
            self.last_solution,
            jac=self.cost_jacobian,
            constraints=self.constraints,
            method='SLSQP',
            options=self.solver_params,
            bounds=self.bounds,
        )
        action = result.x[:self.action_dim]
        if np.isnan(action).any():
            logger.log("WARNING: SLSQP returned nan. Adding noise to last "
                       "action")
            action = self.last_solution[:self.action_dim] + np.random.uniform(
                self.env.action_space.low,
                self.env.action_space.high,
            ) / 100
        else:
            self.last_solution = result.x
        return action, {}
