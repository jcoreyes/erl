import abc
import gym
from gym.spaces import Dict

from railrl.core.distribution import Distribution
from railrl.envs.proxy_env import ProxyEnv


class ContextualRewardFn(object, metaclass=abc.ABCMeta):
    """You can also just pass in a function."""

    @abc.abstractmethod
    def __call__(self, states, actions, next_states, contexts):
        pass


class ContextualEnv(ProxyEnv):
    def __init__(
            self,
            env: gym.Env,
            context_distribution: Distribution,
            reward_fn: ContextualRewardFn,
            context_key='context',
            observation_key='observation',
    ):
        super().__init__(env)
        if not isinstance(env.observation_space, Dict):
            raise ValueError("ContextualEnvs require wrapping Dict spaces.")
        spaces = env.observation_space.spaces
        spaces[context_key] = context_distribution.space
        self.observation_space = Dict(spaces)
        self.context_distribution = context_distribution
        self._reward_fn = reward_fn
        self._context_key = context_key
        self._observation_key = observation_key
        self._last_obs = None
        self.rollout_context = None

    def reset(self):
        obs = self.wrapped_env.reset()
        # TODO: don't assume rollout_context is a vector
        self.rollout_context = self.context_distribution.sample(1)[0]
        self._update_obs(obs)
        self._last_obs = obs
        return obs

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        self._update_obs(obs)
        new_reward = self._compute_reward(self._last_obs, action, obs)
        self._last_obs = obs
        info['ContextualEnv/old_reward'] = reward
        return obs, new_reward, done, info

    def _compute_reward(self, state, action, next_state):
        """Do reshaping for reward_fn, which is implemented for batches."""
        # TODO: don't assume these things are just
        return self._reward_fn(
            state[self._observation_key][None, ...],
            action[None, ...],
            next_state[self._observation_key][None, ...],
            self.rollout_context,
        )[0]

    def _update_obs(self, obs):
        obs[self._context_key] = self.rollout_context
