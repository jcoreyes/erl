from collections import OrderedDict
from random import randint

import tensorflow as tf
import numpy as np

from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.envs.base import Env
from rllab.misc import logger
from sandbox.rocky.tf.spaces.box import Box


def _generate_sign():
    return 2*randint(0, 1) - 1


class HighLow(Env):
    def __init__(self, num_steps, **kwargs):
        assert num_steps > 0
        self._num_steps = num_steps
        self._t = 0
        self._sign = _generate_sign()
        self._action_space = Box(np.array([-1]), np.array([1]))
        self._observation_space = Box(np.array([-1]), np.array([1]))

    @property
    def observation_space(self):
        return self._observation_space

    def reset(self):
        self._t = 0
        self._sign = _generate_sign()
        return np.array([self._sign])

    def step(self, action):
        self._t += 1
        done = self._t == self.horizon
        action = max(-1, min(action, 1))
        if done:
            reward = action * self._sign
        else:
            reward = 0
        observation = np.array([0])
        # To cheat:
        # observation = np.array([self._sign])
        info = self._get_info_dict()
        return observation, reward, done, info

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._num_steps

    def _get_info_dict(self):
        return {
            'target_number': self._sign,
            'time': self._t,
        }

    def get_tf_loss(self, observations, actions, target_labels, **kwargs):
        """
        Return the supervised-learning loss.
        :param observation: Tensor
        :param action: Tensor
        :return: loss Tensor
        """
        target_labels_float = tf.cast(target_labels, tf.float32)
        assert target_labels_float.get_shape().is_compatible_with(
            actions.get_shape()
        )
        return actions * target_labels_float

    def log_diagnostics(self, paths):
        final_values = []
        final_rewards = []
        for path in paths:
            final_value = path["actions"][-1][0]
            final_values.append(final_value)
            final_rewards.append(path["observations"][0][0] * final_value)

        last_statistics = OrderedDict()
        last_statistics.update(create_stats_ordered_dict(
            'Final Value',
            final_values,
        ))
        last_statistics.update(create_stats_ordered_dict(
            'Unclipped Final Rewards',
            final_rewards,
        ))

        for key, value in last_statistics.items():
            logger.record_tabular(key, value)

        return final_rewards

    @staticmethod
    def get_extra_info_dict_from_batch(batch):
        return dict(
            target_numbers=batch['target_numbers'],
            times=batch['times'],
        )

    @staticmethod
    def get_flattened_extra_info_dict_from_subsequence_batch(batch):
        target_numbers = batch['target_numbers']
        times = batch['times']
        flat_target_numbers = target_numbers.flatten()
        flat_times = times.flatten()
        return dict(
            target_numbers=flat_target_numbers,
            times=flat_times,
        )

    @staticmethod
    def get_last_extra_info_dict_from_subsequence_batch(batch):
        target_numbers = batch['target_numbers']
        times = batch['times']
        last_target_numbers = target_numbers[:, -1]
        last_times = times[:, -1]
        return dict(
            target_numbers=last_target_numbers,
            times=last_times,
        )
