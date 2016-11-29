import tensorflow as tf

from core import tf_util
from core.tf_util import he_uniform_initializer
from predictors.state_action_network import StateActionNetwork
from rllab.core.serializable import Serializable


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
        Serializable.quick_init(self, locals())
        self.hidden_sizes = hidden_sizes
        self.hidden_W_init = hidden_W_init or he_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        super(MlpStateActionNetwork, self).__init__(name_or_scope=name_or_scope,
                                                    output_dim=output_dim,
                                                    **kwargs)

    def _create_network(self, observation_input, action_input):
        concat_input = tf.concat(1, [observation_input, action_input])
        hidden_output = tf_util.mlp(
            concat_input,
            self.observation_dim + self.action_dim,
            self.hidden_sizes,
            self.hidden_nonlinearity,
            W_initializer=self.hidden_W_init,
            b_initializer=self.hidden_b_init,
        )
        return self.output_nonlinearity(tf_util.linear(
            hidden_output,
            self.hidden_sizes[-1],
            self.output_dim,
            W_initializer=self.output_W_init,
            b_initializer=self.output_b_init,
        ))