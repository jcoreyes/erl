import abc
import gym
from gym.spaces import Dict
import numpy as np

from railrl.core.distribution import DictDistribution
from railrl import pythonplusplus as ppp


class ContextualRewardFn(object, metaclass=abc.ABCMeta):
    """You can also just pass in a function."""

    @abc.abstractmethod
    def __call__(
            self,
            states: dict,
            actions,
            next_states: dict,
            contexts: dict
    ):
        pass


class ContextualEnv(gym.Wrapper):
    def __init__(
            self,
            env: gym.Env,
            context_distribution: DictDistribution,
            reward_fn: ContextualRewardFn,
            observation_key='observation',
            update_env_info_fn=None,
    ):
        super().__init__(env)
        if not isinstance(env.observation_space, Dict):
            raise ValueError("ContextualEnvs require wrapping Dict spaces.")
        spaces = env.observation_space.spaces
        for k, space in context_distribution.spaces.items():
            spaces[k] = space
        self.observation_space = Dict(spaces)
        self.context_distribution = context_distribution
        self._reward_fn = reward_fn
        self._context_keys = list(context_distribution.spaces.keys())
        self._observation_key = observation_key
        self._last_obs = None
        self._rollout_context_batch = None
        self._update_env_info = update_env_info_fn or insert_reward

    def reset(self):
        obs = self.env.reset()
        self._rollout_context_batch = self.context_distribution.sample(1)
        self._update_obs(obs)
        self._last_obs = obs
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._update_obs(obs)
        new_reward = self._compute_reward(self._last_obs, action, obs)
        self._last_obs = obs
        info = self._update_env_info(self, info, obs, reward, done)
        return obs, new_reward, done, info

    def _compute_reward(self, state, action, next_state):
        """Do reshaping for reward_fn, which is implemented for batches."""
        # TODO: don't assume these things are just vectors
        states = batchify(state)
        actions = batchify(action)
        next_states = batchify(next_state)
        return self._reward_fn(
            states,
            actions,
            next_states,
            self._rollout_context_batch,
        )[0]

    def _update_obs(self, obs):
        for k in self._context_keys:
            obs[k] = self._rollout_context_batch[k][0]


def insert_reward(contexutal_env, info, obs, reward, done):
    info['ContextualEnv/old_reward'] = reward
    return info


def batchify(x):
    return ppp.treemap(lambda x: x[None], x, atomic_type=np.ndarray)
