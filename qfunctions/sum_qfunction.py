import tensorflow as tf

from core.tf_util import weight_variable
from qfunctions.nn_qfunction import NNQFunction


class SumCritic(NNQFunction):
    """Just output the sum of the inputs. This is used to debug."""

    def _create_network(self, observation_input, action_input):
        with tf.variable_scope("actions_layer") as _:
            W_actions = weight_variable(
                (self.action_dim, 1),
                initializer=tf.constant_initializer(1.),
            )
        with tf.variable_scope("observation_layer") as _:
            W_obs = weight_variable(
                (self.observation_dim, 1),
                initializer=tf.constant_initializer(1.),
            )

        return (tf.matmul(action_input, W_actions) +
                tf.matmul(observation_input, W_obs))