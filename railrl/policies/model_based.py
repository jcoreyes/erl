import numpy as np
from torch import nn

from railrl.policies.state_distance import SampleBasedUniversalPolicy
from railrl.torch import pytorch_util as ptu


class GreedyModelBasedPolicy(SampleBasedUniversalPolicy, nn.Module):
    """
    Choose action according to

    a = argmin_a ||f(s, a) - GOAL||^2

    where f is a learned forward dynamics model.

    Do the argmin by sampling a bunch of actions
    """
    def __init__(
            self,
            model,
            env,
            sample_size=100,
            action_penalty=0,
    ):
        super().__init__(sample_size)
        nn.Module.__init__(self)
        self.model = model
        self.env = env
        self.action_penalty = action_penalty

    def get_action(self, obs):
        sampled_actions = self.env.sample_actions(self.sample_size)
        action = ptu.np_to_var(sampled_actions)
        obs = self.expand_np_to_var(obs)
        state_delta_predicted = self.model(
            obs,
            action,
        )
        next_state_predicted = obs + state_delta_predicted
        next_goal_state_predicted = (
            self.env.convert_obs_to_goal_states_pytorch(
                    next_state_predicted
            )
        )
        errors = (next_goal_state_predicted - self._goal_batch)**2
        mean_errors = ptu.get_numpy(errors.mean(dim=1))
        score = mean_errors + self.action_penalty * np.linalg.norm(
            sampled_actions,
            axis=1
        )
        min_i = np.argmin(score)
        return sampled_actions[min_i], {}
