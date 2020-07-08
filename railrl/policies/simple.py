import numpy as np

from railrl.policies.base import Policy


class ZeroPolicy(Policy):
    """
    Policy that always outputs zero.
    """

    def __init__(self, action_dim):
        self.action_dim = action_dim

    def get_action(self, *args, **kwargs):
        return np.zeros(self.action_dim), {}


class RandomPolicy(Policy):
    """
    Policy that always outputs zero.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, *args, **kwargs):
        return self.action_space.sample(), {}

class SamplingPolicy(Policy):
    """
    Policy that samples actions and picks the best action according to the Q function.
    """

    def __init__(self, action_space, qf, base_policy, num_samples=10000):
        self.action_space = action_space
        self.qf = qf
        self.base_policy = base_policy
        self.num_samples = num_samples

    def get_action(self, observation):
        actions = np.random.uniform(
            low=self.action_space.low,
            high=self.action_space.high,
            size=(self.num_samples, len(self.action_space.low)),
        )
        obs = np.repeat(observation[np.newaxis,], self.num_samples, axis=0)
        from railrl.torch.core import torch_ify, np_ify
        v_vals = self.qf(
            torch_ify(obs),
            torch_ify(actions),
        )
        v_vals = np_ify(v_vals)
        best_idx = np.argmax(v_vals)
        return actions[best_idx, :], {}

    def get_actions(self, observations):
        """
        Don't sample actions here, it's too slow when the batch size is large.
        Use the base policy instead.
        """
        return self.base_policy.get_actions(observations)

