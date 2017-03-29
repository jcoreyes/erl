import tensorflow as tf

from railrl.core.tf_util import linear
from railrl.policies.memory.memory_policy import MemoryPolicy


class AffineSoftmaxPolicy(MemoryPolicy):
    """
    write = affine function of environment observation and memory
    logits = affine function of environment observation, memory, and write
    action = softmax(logits)
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

        with tf.variable_scope("write_action"):
            write_action = linear(
                last_layer=env_and_memory,
                last_size=2*self._memory_and_action_dim,
                new_size=self._memory_and_action_dim,
            )

        env_mem_and_write = tf.concat(axis=1, values=[env_obs, memory_obs, write_action])
        with tf.variable_scope("env_action"):
            action_logit = linear(
                last_layer=env_mem_and_write,
                last_size=3*self._memory_and_action_dim,
                new_size=self._memory_and_action_dim,
            )
            env_action = tf.nn.softmax(action_logit, dim=-1)
        return env_action, write_action
