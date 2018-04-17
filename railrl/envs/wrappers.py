import numpy as np
import gym.spaces
import itertools
from gym import Env
from gym.spaces import Box
from scipy.misc import imresize

from railrl.core.serializable import Serializable
from gym.spaces import Discrete


class ProxyEnv(Serializable, Env):
    def __init__(self, wrapped_env):
        Serializable.quick_init(self, locals())
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space
        # self.goal_space = self._wrapped_env.goal_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        if hasattr(self._wrapped_env, 'log_diagnostics'):
            self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()


class ImageEnv(ProxyEnv, Env):
    def __init__(self, wrapped_env, imsize=32):
        self.quick_init(locals())
        super().__init__(wrapped_env)
        self.imsize = imsize
        # change this later
        self.observation_space = Discrete(3*self.imsize*self.imsize)
        #self.i = 0

    def step(self, action):
        _, reward, done, info = super().step(action)
        observation = self._image_observation()
        return observation, reward, done, info

    def reset(self):
        super().reset()
        return self._image_observation()

    def _image_observation(self):
        # image_obs = self.render(mode='rgb_array')
        # downsampled_obs = imresize(image_obs, (self.imsize, self.imsize))
        #fname = 'images/' + str(self.i) + '.png'
        #plt.imsave(fname=fname, arr=downsampled_obs)
        #self.i += 1

        # convert from PIL image format to torch tensor format
        # downsampled_obs = downsampled_obs.transpose((2, 0, 1))
        # return downsampled_obs.flatten()
        return self.wrapped_env.sim.render(self.imsize, self.imsize)



class DiscretizeEnv(ProxyEnv, Env):
    def __init__(self, wrapped_env, num_bins):
        self.quick_init(locals())
        super().__init__(wrapped_env)
        low = self.wrapped_env.action_space.low
        high = self.wrapped_env.action_space.high
        action_ranges = [
            np.linspace(low[i], high[i], num_bins)
            for i in range(len(low))
        ]
        self.idx_to_continuous_action = [
            np.array(x) for x in itertools.product(*action_ranges)
        ]
        self.action_space = Discrete(len(self.idx_to_continuous_action))

    def step(self, action):
        continuous_action = self.idx_to_continuous_action[action]
        return super().step(continuous_action)



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
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub, dtype=np.float32)

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


"""
Some wrapper codes for rllab.
"""


class ConvertEnvToRllab(ProxyEnv, Serializable):
    """
    rllab sometimes requires the action/observation space to be a specific type
    """
    def __init__(self, env):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self.action_space = convert_space_to_rllab_space(
            self._wrapped_env.action_space
        )
        self.observation_space = convert_space_to_rllab_space(
            self._wrapped_env.observation_space
        )
        from rllab.envs.env_spec import EnvSpec
        self.spec = EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

    def __str__(self):
        return "RllabConverted: %s" % self._wrapped_env

    def get_param_values(self):
        if hasattr(self.wrapped_env, "get_param_values"):
            return self.wrapped_env.get_param_values()
        return None


def convert_space_to_rllab_space(space):
    from sandbox.rocky.tf.spaces import Box as TfBox
    from sandbox.rocky.tf.spaces import Discrete as TfDiscrete
    from rllab.spaces.discrete import Discrete as RllabDiscrete
    from rllab.spaces.box import Box as RllabBox
    if isinstance(space, RllabBox) or isinstance(space, RllabDiscrete):
        return space
    if isinstance(space, TfBox) or isinstance(space, gym.spaces.Box):
        return RllabBox(space.low, space.high)
    elif isinstance(space, TfDiscrete) or isinstance(space, gym.spaces.Discrete):
        return RllabDiscrete(space.n)
    raise TypeError()


class ConvertEnvToTf(ProxyEnv, Serializable):
    def __init__(self, env):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self.action_space = convert_space_to_tf_space(
            self._wrapped_env.action_space
        )
        self.observation_space = convert_space_to_tf_space(
            self._wrapped_env.observation_space
        )
        from rllab.envs.env_spec import EnvSpec
        self.spec = EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

    def __str__(self):
        return "TfConverted: %s" % self._wrapped_env

    def get_param_values(self):
        if hasattr(self.wrapped_env, "get_param_values"):
            return self.wrapped_env.get_param_values()
        return None


def convert_space_to_tf_space(space):
    from sandbox.rocky.tf.spaces import Box as TfBox
    from sandbox.rocky.tf.spaces import Discrete as TfDiscrete
    from rllab.spaces.discrete import Discrete as RllabDiscrete
    from rllab.spaces.box import Box as RllabBox
    if isinstance(space, TfBox) or isinstance(space, TfDiscrete):
        return space
    if isinstance(space, RllabBox) or isinstance(space, gym.spaces.Box):
        return TfBox(space.low, space.high)
    elif isinstance(space, RllabDiscrete) or isinstance(space, gym.spaces.Discrete):
        return TfDiscrete(space.n)
    raise TypeError()
