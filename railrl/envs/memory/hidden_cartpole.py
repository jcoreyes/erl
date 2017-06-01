from collections import OrderedDict

from cached_property import cached_property
import numpy as np

from railrl.envs.env_utils import gym_env
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import split_paths
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import convert_gym_space, GymEnv
from rllab.envs.normalized_env import normalize, NormalizedEnv
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc import logger
from sandbox.rocky.tf.spaces import Box


class HiddenCartpoleEnv(CartpoleEnv, Serializable):
    def __init__(self, num_steps=100, position_only=True):
        Serializable.quick_init(self, locals())
        assert position_only, "I only added position_only due to some weird " \
                              "serialization bug"
        CartpoleEnv.__init__(self, position_only=position_only)
        self.num_steps = num_steps

    @cached_property
    def action_space(self):
        return Box(super().action_space.low,
                   super().action_space.high)

    @cached_property
    def observation_space(self):
        return Box(super().observation_space.low,
                   super().observation_space.high)

    @property
    def horizon(self):
        return self.num_steps

    @staticmethod
    def get_extra_info_dict_from_batch(batch):
        return dict()

    @staticmethod
    def get_flattened_extra_info_dict_from_subsequence_batch(batch):
        return dict()

    @staticmethod
    def get_last_extra_info_dict_from_subsequence_batch(batch):
        return dict()

    def log_diagnostics(self, paths, **kwargs):
        list_of_rewards, terminals, obs, actions, next_obs = split_paths(paths)

        returns = []
        for rewards in list_of_rewards:
            returns.append(np.sum(rewards))
        last_statistics = OrderedDict()
        last_statistics.update(create_stats_ordered_dict(
            'UndiscountedReturns',
            returns,
        ))
        last_statistics.update(create_stats_ordered_dict(
            'Actions',
            actions,
        ))

        for key, value in last_statistics.items():
            logger.record_tabular(key, value)
        return returns

    def is_current_done(self):
        return False


class NormalizedHiddenCartpoleEnv(ProxyEnv):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        env = HiddenCartpoleEnv(*args, **kwargs)
        env = normalize_tf(env)
        super().__init__(env)

    @staticmethod
    def get_extra_info_dict_from_batch(batch):
        return dict()

    @staticmethod
    def get_flattened_extra_info_dict_from_subsequence_batch(batch):
        return dict()

    @staticmethod
    def get_last_extra_info_dict_from_subsequence_batch(batch):
        return dict()

    def log_diagnostics(self, paths, **kwargs):
        return self._wrapped_env.log_diagnostics(paths, **kwargs)


class NormalizedTfEnv(NormalizedEnv):
    @property
    def action_space(self):
        return Box(super().action_space.low,
                   super().action_space.high)

    @property
    def observation_space(self):
        return Box(super().observation_space.low,
                   super().observation_space.high)

    def log_diagnostics(self, paths, **kwargs):
        return self._wrapped_env.log_diagnostics(paths, **kwargs)

normalize_tf = NormalizedTfEnv


class ConvertEnv(ProxyEnv, Serializable):
    def __init__(self, env):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

    @property
    def action_space(self):
        return Box(super().action_space.low,
                   super().action_space.high)

    @property
    def observation_space(self):
        return Box(super().observation_space.low,
                   super().observation_space.high)

    def __str__(self):
        return "TfConverted: %s" % self._wrapped_env


convert_to_tf_env = ConvertEnv


