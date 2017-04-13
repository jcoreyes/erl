from railrl.exploration_strategies.base import RawExplorationStrategy
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
import numpy as np
import numpy.random as nr


class OUStrategy(RawExplorationStrategy, Serializable):
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process

    Based on the rllab implementation.
    """

    def __init__(self, env_spec, mu=0, theta=0.15, sigma=0.3, **kwargs):
        assert isinstance(env_spec.action_space, Box)
        assert len(env_spec.action_space.shape) == 1
        Serializable.quick_init(self, locals())
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_space = env_spec.action_space
        self.state = np.ones(self.action_space.flat_dim) * self.mu
        self.reset()

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["state"] = self.state
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.state = d["state"]

    def reset(self):
        self.state = np.ones(self.action_space.flat_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        # TODO check if decaying exploration noise helps
        # self.sigma *= 0.99
        return self.state

    def get_action(self, t, observation, policy, **kwargs):
        action, agent_info = policy.get_action(observation)
        return self.get_action_from_raw_action(action), agent_info

    def get_action_from_raw_action(self, action, **kwargs):
        ou_state = self.evolve_state()
        return np.clip(action + ou_state, self.action_space.low, self.action_space.high)
