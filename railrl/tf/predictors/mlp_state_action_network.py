import tensorflow as tf

from railrl.tf.core import tf_util
from railrl.tf.core.tf_util import he_uniform_initializer
from railrl.tf.predictors.state_action_network import StateActionNetwork


class MlpStateActionNetwork(StateActionNetwork):
    def __init__(
            self,
            name_or_scope,
            output_dim,
            hidden_sizes=(100, 100),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.identity,
            **kwargs
    ):
        self.setup_serialization(locals())
        super(MlpStateActionNetwork, self).__init__(name_or_scope=name_or_scope,
                                                    output_dim=output_dim,
                                                    **kwargs)
        self.hidden_sizes = hidden_sizes
        self.hidden_W_init = hidden_W_init or he_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self._create_network()

    def _create_network_internal(self, observation_input, action_input):
        observation_input = self._process_layer(observation_input,
                                                scope_name="observation_input")
        action_input = self._process_layer(action_input,
                                           scope_name="action_input")
        concat_input = tf.concat(axis=1, values=[observation_input, action_input])
        hidden_output = tf_util.mlp(
            concat_input,
            self.observation_dim + self.action_dim,
            self.hidden_sizes,
            self.hidden_nonlinearity,
            W_initializer=self.hidden_W_init,
            b_initializer=self.hidden_b_init,
            pre_nonlin_lambda=self._process_layer,
        )
        return self.output_nonlinearity(tf_util.linear(
            hidden_output,
            self.hidden_sizes[-1],
            self.output_dim,
            W_initializer=self.output_W_init,
            b_initializer=self.output_b_init,
        ))
