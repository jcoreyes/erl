import numpy as np
import tensorflow as tf

from railrl.policies.memory.memory_policy import MemoryPolicy
from railrl.tf.core.tf_util import weight_variable


class LinearOcmPolicy(MemoryPolicy):
    """
    A custom linear policy for solving the one character memory task.
    Used as a sanity check.

    Note: this policy does not support batch norm and other between-layer
    operations. (To add this, add _process_layer_ calls in
    _create_network_internal.)
    """

    def __init__(
            self,
            name_or_scope,
            memory_and_action_dim,
            **kwargs
    ):
        self.setup_serialization(locals())
        self._memory_and_action_dim = memory_and_action_dim
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network_internal(self, observation_input=None):
        assert observation_input is not None
        env_obs, memory_obs = observation_input
        env_and_memory = tf.concat(axis=1, values=[env_obs, memory_obs])

        # Initially, make write_action = env_obs[1:] + memory_obs
        all_but_last = np.eye(self._memory_and_action_dim)
        all_but_last[0] = 0
        arr = np.vstack((
            all_but_last,
            np.eye(self._memory_and_action_dim),
        ))
        sum_matrix = weight_variable(
            [2 * self._memory_and_action_dim, self._memory_and_action_dim],
            initializer=tf.constant_initializer(arr),
            name="combine_matrix",
            regularizable=True,
        )
        write_action = tf.matmul(env_and_memory, sum_matrix)

        # Initially, make env_action = write_action
        arr2 = np.vstack((
            np.zeros((self._memory_and_action_dim, self._memory_and_action_dim)),
            np.zeros((self._memory_and_action_dim, self._memory_and_action_dim)),
            np.eye(self._memory_and_action_dim),
        ))
        env_action_matrix = weight_variable(
            [3 * self._memory_and_action_dim, self._memory_and_action_dim],
            initializer=tf.constant_initializer(arr2),
            name="remove_last_value_matrix",
            regularizable=True,
        )
        env_memory_and_write = tf.concat(axis=1, values=[env_and_memory, write_action])
        env_action = tf.matmul(env_memory_and_write, env_action_matrix)
        env_action = tf.nn.relu(env_action)
        env_action /= tf.reduce_sum(env_action, 1, keep_dims=True)

        return env_action, write_action
