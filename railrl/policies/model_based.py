import numpy as np
import torch
from torch import nn
from scipy import optimize

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
    ):
        super().__init__()
        nn.Module.__init__(self)
        self.model = model
        self.env = env
        self.model_learns_deltas = model_learns_deltas
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
        self.bounds = list(zip(self.lower_bounds, self.upper_bounds))
        self.constraints = {
            'type': 'eq',
            'fun': self.constraint_fctn,
            'jac': self.constraint_jacobian,
        }

    def cost_function(self, action_and_next_state_flat):
        next_state = action_and_next_state_flat[self.action_dim:]
        return np.sum((next_state - self._goal_np)**2)

    def cost_jacobian(self, action_and_next_state_flat):
        next_state = action_and_next_state_flat[self.action_dim:]
        jac = 2 * (next_state - self._goal_np)
        newjac = np.hstack((np.zeros(self.action_dim), jac))
        return newjac

    def constraint_fctn(self, action_next_state_flat, state=None):
        state = ptu.np_to_var(state)
        action_next_state_flat = ptu.np_to_var(
            action_next_state_flat,
            requires_grad=False,
        )
        action = action_next_state_flat[0:self.action_dim]
        next_state = action_next_state_flat[self.action_dim:]

        if self.model_learns_deltas:
            next_state_predicted = state + self.model(
                state.unsqueeze(0),
                action.unsqueeze(0),
            )
        else:
            next_state_predicted = self.model(
                state.unsqueeze(0),
                action.unsqueeze(0),
            )
        loss = torch.norm(next_state - next_state_predicted, p=2)
        return ptu.get_numpy(loss)

    def constraint_jacobian(self, action_next_state_flat, state=None):
        state = ptu.np_to_var(state)
        action_next_state_flat = ptu.np_to_var(
            action_next_state_flat,
            requires_grad=True,
        )
        action = action_next_state_flat[0:self.action_dim]
        next_state = action_next_state_flat[self.action_dim:]

        if self.model_learns_deltas:
            next_state_predicted = state + self.model(
                state.unsqueeze(0),
                action.unsqueeze(0),
            )
        else:
            next_state_predicted = self.model(
                state.unsqueeze(0),
                action.unsqueeze(0),
            )
        loss = torch.norm(next_state - next_state_predicted, p=2)
        loss.squeeze(0).backward()
        return ptu.get_numpy(action_next_state_flat.grad)

    def reset(self):
        self.last_solution = np.zeros(self.action_dim + self.observation_dim)

    def get_action(self, obs):
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
