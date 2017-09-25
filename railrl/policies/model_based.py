import numpy as np
from torch import nn

from railrl.policies.state_distance import SampleBasedUniversalPolicy
from railrl.torch import pytorch_util as ptu


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
        super().__init__(sample_size)
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
