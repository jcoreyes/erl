import numpy as np
import gym.spaces
from cached_property import cached_property

from railrl.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from sandbox.rocky.tf.spaces import Box as TfBox
from sandbox.rocky.tf.spaces import Discrete as TfDiscrete
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product


class NormalizedBoxEnv(ProxyEnv, Serializable):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """
    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_mean=None,
            obs_std=None,
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        # Add these explicitly in case they were modified
        d["_obs_mean"] = self._obs_mean
        d["_obs_std"] = self._obs_std
        d["_reward_scale"] = self._reward_scale
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._obs_mean = d["_obs_mean"]
        self._obs_std = d["_obs_std"]
        self._reward_scale = d["_reward_scale"]

    @property
    def action_space(self):
        ub = np.ones(self._wrapped_env.action_space.shape)
        return Box(-1 * ub, ub)

    @property
    def observation_space(self):
        return Box(super().observation_space.low,
                   super().observation_space.high)

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    def log_diagnostics(self, paths, **kwargs):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            return self._wrapped_env.log_diagnostics(paths, **kwargs)
        else:
            return None

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)

normalize_box = NormalizedBoxEnv


class ConvertEnvToTf(ProxyEnv, Serializable):
    def __init__(self, env):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

    @property
    def action_space(self):
        action_space = self._wrapped_env.action_space
        if isinstance(action_space, TfBox) or isinstance(action_space,
                                                         TfDiscrete):
            return action_space
        if isinstance(action_space, Box) or isinstance(action_space, gym.spaces.Box):
            return TfBox(action_space.low, action_space.high)
        elif isinstance(action_space, Discrete) or isinstance(action_space, gym.spaces.Discrete):
            return TfDiscrete(action_space.n)
        raise TypeError()

    @property
    def observation_space(self):
        return TfBox(super().observation_space.low,
                     super().observation_space.high)

    def __str__(self):
        return "TfConverted: %s" % self._wrapped_env

    def get_param_values(self):
        if hasattr(self.wrapped_env, "get_param_values"):
            return self.wrapped_env.get_param_values()
        return None

    def log_diagnostics(self, paths, *args, **kwargs):
        if hasattr(self.wrapped_env, "log_diagnostics"):
            self.wrapped_env.log_diagnostics(paths, *args, **kwargs)

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()


convert_to_tf_env = ConvertEnvToTf


class NormalizeAndConvertToTfEnv(NormalizedBoxEnv, ConvertEnvToTf):
    @property
    def action_space(self):
        # Apparently this is how you call a super's property
        return ConvertEnvToTf.action_space.fget(self)

    @property
    def observation_space(self):
        return TfBox(super().observation_space.low,
                     super().observation_space.high)

    def __str__(self):
        return "TfNormalizedAndConverted: %s" % self._wrapped_env

normalize_and_convert_to_tf_env = NormalizeAndConvertToTfEnv


class ConvertEnvToRllab(ProxyEnv, Serializable):
    """
    rllab sometimes requires the action/observation space to be a specific type
    """
    def __init__(self, env):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

    @cached_property
    def action_space(self):
        action_space = self._wrapped_env.action_space
        if isinstance(action_space, Box) or isinstance(action_space, Discrete):
            return action_space
        if isinstance(action_space, TfBox) or isinstance(action_space, gym.spaces.Box):
            return Box(action_space.low, action_space.high)
        elif isinstance(action_space, TfDiscrete) or isinstance(action_space, gym.spaces.Discrete):
            return Discrete(action_space.n)
        raise TypeError()

    @cached_property
    def observation_space(self):
        return Box(super().observation_space.low, super().observation_space.high)

    def __str__(self):
        return "RllabConverted: %s" % self._wrapped_env

    def get_param_values(self):
        if hasattr(self.wrapped_env, "get_param_values"):
            return self.wrapped_env.get_param_values()
        return None

    def log_diagnostics(self, paths, *args, **kwargs):
        if hasattr(self.wrapped_env, "log_diagnostics"):
            self.wrapped_env.log_diagnostics(paths, *args, **kwargs)

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()


def convert_gym_space(space):
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return Product([convert_gym_space(x) for x in space.spaces])
    elif (isinstance(space, Box) or isinstance(space, Discrete)
          or isinstance(space, Product)):
        return space
    else:
        raise NotImplementedError
