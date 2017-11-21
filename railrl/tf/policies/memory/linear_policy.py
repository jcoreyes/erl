import tensorflow as tf

from railrl.tf.core.tf_util import weight_variable
from railrl.tf.policies.memory.memory_policy import MemoryPolicy


class LinearPolicy(MemoryPolicy):
    """
    A linear policy based on LinearOcmPolicy. Basically

    write_action = A * [env; memory]
    env_action = B * [env; memory; write_action]

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

        sum_matrix = weight_variable(
            [2 * self._memory_and_action_dim, self._memory_and_action_dim],
            name="combine_matrix",
            regularizable=True,
        )
        write_action = tf.matmul(env_and_memory, sum_matrix)

        env_action_matrix = weight_variable(
            [3 * self._memory_and_action_dim, self._memory_and_action_dim],
            name="remove_last_value_matrix",
            regularizable=True,
        )
        env_memory_and_write = tf.concat(axis=1, values=[env_and_memory, write_action])
        env_action = tf.matmul(env_memory_and_write, env_action_matrix)
        env_action = tf.nn.relu(env_action)
        env_action = tf.nn.l2_normalize(env_action, dim=1)

        return env_action, write_action
