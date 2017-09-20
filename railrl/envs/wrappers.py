import numpy as np
import gym.spaces
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from sandbox.rocky.tf.spaces import Box as TfBox
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product


class NormalizedBoxEnv(ProxyEnv, Serializable):
    """
    Normalize action to in [0, 1].
    Optionally normalize STD.
    """
    def __init__(
            self,
            env,
            scale_reward=1.,
            obs_mean=None,
            obs_std=None,
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self._wrapped_env = env
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
        self._scale_reward = scale_reward
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
        d["_scale_reward"] = self._scale_reward
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._obs_mean = d["_obs_mean"]
        self._obs_std = d["_obs_std"]
        self._scale_reward = d["_scale_reward"]

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
        reward *= self._scale_reward
        return next_obs, reward * self._scale_reward, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    def log_diagnostics(self, paths, **kwargs):
        return self._wrapped_env.log_diagnostics(paths, **kwargs)

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)

normalize_box = NormalizedBoxEnv


class ConvertEnv(ProxyEnv, Serializable):
    def __init__(self, env):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

    @property
    def action_space(self):
        return TfBox(super().action_space.low,
                   super().action_space.high)

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


convert_to_tf_env = ConvertEnv


class NormalizeAndConvertEnv(NormalizedBoxEnv, ConvertEnv):
    @property
    def action_space(self):
        return TfBox(super().action_space.low,
                     super().action_space.high)

    @property
    def observation_space(self):
        return TfBox(super().observation_space.low,
                     super().observation_space.high)

    def __str__(self):
        return "TfNormalizedAndConverted: %s" % self._wrapped_env

normalize_and_convert_to_tf_env = ConvertEnv


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
