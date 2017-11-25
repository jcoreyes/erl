import tensorflow as tf

from railrl.qfunctions.nn_qfunction import NNQFunction
from railrl.tf.core.tf_util import linear


class AffineTanHQFunction(NNQFunction):
    """
    Q = TanH(affine(inputs))
    """

    def __init__(
            self,
            name_or_scope,
            **kwargs
    ):
        self.setup_serialization(locals())
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network_internal(
            self,
            observation_input=None,
            action_input=None,
    ):
        env_obs, memory_obs = observation_input
        env_action, memory_action = action_input
        all_input = tf.concat(
            axis=1,
            values=[env_obs, memory_obs, env_action, memory_action]
        )
        with tf.variable_scope("output_linear"):
            return tf.nn.tanh(linear(
                all_input,
                sum(self.observation_dim) + sum(self.action_dim),
                1,
            ))
