import tensorflow as tf

from core import tf_util
from predictors.mlp_state_network import MlpStateNetwork
from qfunctions.nn_qfunction import NNQFunction
from rllab.core.serializable import Serializable


class QuadraticQF(NNQFunction):
    def __init__(
            self,
            name_or_scope,
            policy,
            reuse=False,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        self.policy = policy
        super(QuadraticQF, self).__init__(
            name_or_scope=name_or_scope,
            **kwargs
        )

    def _create_network(self, observation_input, action_input):
        L_params = MlpStateNetwork(
            name_or_scope="L",
            output_dim=self.action_dim * self.action_dim,
            observation_dim=self.observation_dim,
            observation_input=observation_input,
            observation_hidden_sizes=(200, 200),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.identity,
            )
        # L_shape = batch:dimA:dimA
        L = tf_util.vec2lower_triangle(L_params.output, self.action_dim)

        delta = action_input - self.policy.output
        h1 = tf.expand_dims(delta, 1)  # h1_shape = batch:1:dimA
        h1 = tf.batch_matmul(h1, L)    # h1_shape = batch:1:dimA
        h1 = tf.batch_matmul(
            h1,
            h1,
            adj_y=True,  # Compute h1 * h1^T
        )                              # h1_shape = batch:1:1
        h1 = tf.squeeze(h1, [1])       # h1_shape = batch:1
        return -0.5 * h1
