import numpy as np
import tensorflow as tf

from railrl.core.tf_util import weight_variable
from railrl.policies.memory_policy import MemoryPolicy


class LinearOcmPolicy(MemoryPolicy):
    """
    A custom linear policy for solving the one character memory task.
    Used as a sanity check.
    """

    def __init__(
            self,
            name_or_scope,
            memory_and_action_dim,
            horizon,
            **kwargs
    ):
        self.setup_serialization(locals())
        self._action_dim = memory_and_action_dim
        self._horizon = horizon
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network_internal(self, observation_input=None):
        assert observation_input is not None
        env_obs, memory_obs = observation_input

        # Initially, make write_action = env_obs[:-1] + memory_obs[:-1]
        all_but_last = np.eye(self._action_dim)
        all_but_last[-1] = 0
        arr = np.vstack((all_but_last, all_but_last))
        sum_matrix = weight_variable(
            [2 * self._action_dim, self._action_dim],
            initializer=tf.constant_initializer(arr),
            name="combine_matrix",
            regularizable=True,
        )
        env_and_memory = tf.concat(1, [env_obs, memory_obs])
        write_action = tf.matmul(env_and_memory, sum_matrix)

        # This will make it so that write_action = stored_value
        # the last value will be replaced with a zero.
        copy_all_but_last = np.eye(self._action_dim)
        copy_all_but_last[-1] = 0
        remove_last_value = weight_variable(
            [self._action_dim, self._action_dim],
            initializer=tf.constant_initializer(copy_all_but_last),
            name="remove_last_value_matrix",
            regularizable=True,
        )
        stored_value = tf.matmul(write_action, remove_last_value)

        # np_zero_onehot = np.zeros(self._action_dim)
        # np_zero_onehot[0] = 1
        # zero_onehot = weight_variable(
        #     [None, self._action_dim],
        #     initializer=tf.constant_initializer(np_zero_onehot),
        #     name="zero_onehot",
        # )

        time = write_action[:, 0]
        is_last_time_step = tf.cast(time >= self._horizon, tf.float32)

        # env_action = tf.select(is_last_time_step, stored_value, zero_onehot)
        env_action = tf.mul(is_last_time_step, stored_value)

        return env_action, write_action
