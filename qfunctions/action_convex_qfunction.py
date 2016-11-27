import tensorflow as tf

from core.tf_util import mlp, linear, he_uniform_initializer
from qfunctions.nn_qfunction import NNQFunction
from rllab.core.serializable import Serializable


class ActionConvexQFunction(NNQFunction):
    def __init__(
            self,
            name_or_scope,
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            embedded_hidden_sizes=(100,),
            observation_hidden_sizes=(100,),
            hidden_nonlinearity=tf.nn.relu,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        self.hidden_W_init = hidden_W_init or he_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.embedded_hidden_sizes = embedded_hidden_sizes
        self.observation_hidden_sizes = observation_hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network(self, observation_input, action_input):
        with tf.variable_scope("observation_mlp") as _:
            observation_output = mlp(observation_input,
                                     self.observation_dim,
                                     self.observation_hidden_sizes,
                                     self.hidden_nonlinearity,
                                     W_initializer=self.hidden_W_init,
                                     b_initializer=self.hidden_b_init,
                                     reuse_variables=True)
        embedded = tf.concat(1, [observation_output, action_input])
        embedded_dim = self.action_dim + self.observation_hidden_sizes[-1]
        with tf.variable_scope("fusion_mlp") as action_input_scope:
            fused_output = mlp(embedded,
                               embedded_dim,
                               self.embedded_hidden_sizes,
                               self.hidden_nonlinearity,
                               W_initializer=self.hidden_W_init,
                               b_initializer=self.hidden_b_init,
                               reuse_variables=True)

            with tf.variable_scope("output_linear") as _:
                output = linear(fused_output,
                                self.embedded_hidden_sizes[-1],
                                1,
                                W_initializer=self.output_W_init,
                                b_initializer=self.output_b_init,
                                reuse_variables=True)
            self.action_input_scope_name = (
                action_input_scope.original_name_scope)
        return output

    def get_action_params(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 self.action_input_scope_name)
