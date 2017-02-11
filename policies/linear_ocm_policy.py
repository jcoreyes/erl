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
        env_and_memory = tf.concat(1, [env_obs, memory_obs])

        # Initially, make write_action = env_obs[1:] + memory_obs
        all_but_last = np.eye(self._action_dim)
        all_but_last[0] = 0
        arr = np.vstack((
            all_but_last,
            np.eye(self._action_dim),
        ))
        sum_matrix = weight_variable(
            [2 * self._action_dim, self._action_dim],
            initializer=tf.constant_initializer(arr),
            name="combine_matrix",
            regularizable=True,
        )
        write_action = tf.matmul(env_and_memory, sum_matrix)

        # Initially, make env_action = memory_obs
        arr2 = np.vstack((
            np.zeros((self._action_dim, self._action_dim)),
            np.eye(self._action_dim),
        ))
        env_action_matrix = weight_variable(
            [2 * self._action_dim, self._action_dim],
            initializer=tf.constant_initializer(arr2),
            name="remove_last_value_matrix",
            regularizable=True,
        )
        env_action = tf.matmul(env_and_memory, env_action_matrix)

        return env_action, write_action
