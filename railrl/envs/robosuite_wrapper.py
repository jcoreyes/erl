from collections import OrderedDict

from gym import Env
from gym.spaces import Box, Dict
import numpy as np
import robosuite as suite
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict

"""
Based on rllab's serializable.py file
https://github.com/rll/rllab
"""

import inspect
import sys


class Serializable(object):

    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def quick_init(self, locals_):
        if getattr(self, "_serializable_initialized", False):
            return
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(self.__init__)
            # Exclude the first "self" parameter
            if spec.varkw:
                kwargs = locals_[spec.varkw].copy()
            else:
                kwargs = dict()
            if spec.kwonlyargs:
                for key in spec.kwonlyargs:
                    kwargs[key] = locals_[key]
        else:
            spec = inspect.getargspec(self.__init__)
            if spec.keywords:
                kwargs = locals_[spec.keywords]
            else:
                kwargs = dict()
        if spec.varargs:
            varargs = locals_[spec.varargs]
        else:
            varargs = tuple()
        in_order_args = [locals_[arg] for arg in spec.args][1:]
        self.__args = tuple(in_order_args) + varargs
        self.__kwargs = kwargs
        setattr(self, "_serializable_initialized", True)

    def __getstate__(self):
        return {"__args": self.__args, "__kwargs": self.__kwargs}

    def __setstate__(self, d):
        # convert all __args to keyword-based arguments
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(self.__init__)
        else:
            spec = inspect.getargspec(self.__init__)
        in_order_args = spec.args[1:]
        out = type(self)(**dict(zip(in_order_args, d["__args"]), **d["__kwargs"]))
        self.__dict__.update(out.__dict__)

    @classmethod
    def clone(cls, obj, **kwargs):
        assert isinstance(obj, Serializable)
        d = obj.__getstate__()
        d["__kwargs"] = dict(d["__kwargs"], **kwargs)
        out = type(obj).__new__(type(obj))
        out.__setstate__(d)
        return out

class RobosuiteStateWrapperEnv(Serializable, Env):
    def __init__(self, wrapped_env_id, **wrapped_env_kwargs):
        Serializable.quick_init(self, locals())
        self._wrapped_env = suite.make(
            wrapped_env_id,
            **wrapped_env_kwargs
        )
        self.action_space = Box(self._wrapped_env.action_spec[0], self._wrapped_env.action_spec[1], dtype=np.float32)
        observation_dim = self._wrapped_env.observation_spec()['robot-state'].shape[0] \
                          + self._wrapped_env.observation_spec()['object-state'].shape[0]
        self.observation_space = Box(
            -np.inf * np.ones(observation_dim),
            np.inf * np.ones(observation_dim),
            dtype=np.float32,
        )

    def step(self, action):
        obs, reward, done, info = self._wrapped_env.step(action)
        obs = self.flatten_dict_obs(obs)
        return obs, reward, done, info

    def flatten_dict_obs(self, obs):
        robot_state = obs['robot-state']
        object_state = obs['object-state']
        obs = np.concatenate((robot_state, object_state))
        return obs

    def reset(self):
        obs = self._wrapped_env.reset()
        obs = self.flatten_dict_obs(obs)
        return obs

    def render(self):
        self._wrapped_env.render()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)

