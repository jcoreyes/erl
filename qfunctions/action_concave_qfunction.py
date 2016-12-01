import tensorflow as tf

from core.tf_util import mlp, linear, he_uniform_initializer
from policies.argmax_policy import ArgmaxPolicy
from policies.nn_policy import FeedForwardPolicy
from qfunctions.nn_qfunction import NNQFunction
from qfunctions.optimizable_q_function import OptimizableQFunction
from qfunctions.quadratic_qf import QuadraticQF


class ActionConcaveQFunction(NNQFunction, OptimizableQFunction):
    def __init__(
            self,
            name_or_scope,
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            embedded_hidden_sizes=(20, 20, 20),
            observation_hidden_sizes=(100, 100),
            **kwargs
    ):
        self.setup_serialization(locals())
        self.hidden_W_init = hidden_W_init or he_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            0., 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.embedded_hidden_sizes = embedded_hidden_sizes
        self.observation_hidden_sizes = observation_hidden_sizes
        self.hidden_nonlinearity = tf.nn.relu
        self._policy = None
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network(self, observation_input, action_input):
        with tf.variable_scope("observation_mlp") as _:
            observation_output = mlp(observation_input,
                                     self.observation_dim,
                                     self.observation_hidden_sizes,
                                     self.hidden_nonlinearity,
                                     W_initializer=self.hidden_W_init,
                                     b_initializer=self.hidden_b_init,
                                     )
        embedded = tf.concat(1, [observation_output, action_input])
        embedded_dim = self.action_dim + self.observation_hidden_sizes[-1]
        with tf.variable_scope("fusion_mlp") as action_input_scope:
            fused_output = mlp(embedded,
                               embedded_dim,
                               self.embedded_hidden_sizes,
                               self.hidden_nonlinearity,
                               W_initializer=self.hidden_W_init,
                               b_initializer=self.hidden_b_init,
                               )

            with tf.variable_scope("output_linear") as _:
                output = linear(fused_output,
                                self.embedded_hidden_sizes[-1],
                                1,
                                W_initializer=self.output_W_init,
                                b_initializer=self.output_b_init,
                                )
            self.action_input_scope_name = (
                action_input_scope.original_name_scope)
        return -output
        # self.quad_policy = FeedForwardPolicy(
        #     name_or_scope="mu",
        #     action_dim=self.action_dim,
        #     observation_dim=self.observation_dim,
        #     observation_input=observation_input,
        #     observation_hidden_sizes=(200, 200),
        #     hidden_W_init=None,
        #     hidden_b_init=None,
        #     output_W_init=None,
        #     output_b_init=None,
        #     hidden_nonlinearity=tf.nn.relu,
        #     output_nonlinearity=tf.nn.tanh,
        # )
        # self._af = QuadraticQF(
        #     name_or_scope="advantage_function",
        #     action_input=action_input,
        #     observation_input=observation_input,
        #     action_dim=self.action_dim,
        #     observation_dim=self.observation_dim,
        #     policy=self.quad_policy,
        # )
        # return self._af.output

    def get_action_W_params(self):
        """
        Return params that should be clipped to non-negative
        :return:
        """
        return [param
                for param in tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    self.action_input_scope_name)
                if 'bias' not in param.name]

    @property
    def implicit_policy(self):
        if self._policy is None:
            with self.sess.as_default():
                self.sess.run(tf.initialize_variables(self.get_params()))
                self._policy = ArgmaxPolicy(
                    name_or_scope="argmax_policy",
                    qfunction=self,
                )
        return self._policy
