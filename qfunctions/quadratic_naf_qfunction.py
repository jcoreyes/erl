import tensorflow as tf

from misc.rllab_util import get_observation_dim
from policies.nn_policy import FeedForwardPolicy
from predictors.mlp_state_network import MlpStateNetwork
from qfunctions.naf_qfunction import NAFQFunction
from qfunctions.quadratic_qf import QuadraticQF
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides


class QuadraticNAF(NAFQFunction):
    def __init__(
            self,
            name_or_scope,
            observation_input=None,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        observation_dim = get_observation_dim(**kwargs)
        observation_placeholder = tf.placeholder(tf.float32,
                                                 shape=[None, observation_dim])
        super(NAFQFunction, self).__init__(
            name_or_scope=name_or_scope,
            observation_input=observation_placeholder,
            **kwargs
        )

    @overrides
    def _create_network(self, observation_input, action_input):
        self.policy = FeedForwardPolicy(
            name_or_scope="mu",
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
            observation_input=observation_input,
            observation_hidden_sizes=(200, 200),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
        )
        self.vf = MlpStateNetwork(
            name_or_scope="V_function",
            output_dim=1,
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
        self.af = QuadraticQF(
            name_or_scope="advantage_function",
            action_input=action_input,
            observation_input=observation_input,
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
            policy=self.policy,
        )
        return self.vf.output + self.af.output

    def get_implicit_policy(self):
        return self.policy

    def get_implicit_value_function(self):
        return self.vf

    def get_implicit_advantage_function(self):
        return self.af
