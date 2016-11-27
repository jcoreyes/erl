import tensorflow as tf

from core.neuralnet import NeuralNetwork
from misc.rllab_util import get_observation_dim
from policies.nn_policy import NNPolicy
from predictors.mlp_state_network import MlpStateNetwork
from qfunctions.action_convex_qfunction import ActionConvexQFunction
from qfunctions.naf_qfunction import NAFQFunction
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.policies.base import Policy


class ConvexNAF(NAFQFunction):
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
        # TODO(vpong): fix this. Needed for serialization to work
        self.observation_input = self.policy.observation_input
        observation_input = self.observation_input
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
        self.af = ActionConvexQFunction(
            name_or_scope="advantage_function",
            action_input=action_input,
            observation_input=observation_input,
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
            policy=self.policy,
        )
        self.policy = ArgmaxPolicy(self.af)
        return self.vf.output + self.af.output

    def create_implicit_policy(self, action_convex_qfunct):
        pass

    def get_implicit_policy(self):
        return self.policy

    def get_implicit_value_function(self):
        return self.vf

    def get_implicit_advantage_function(self):
        return self.af


class ArgmaxPolicy(NeuralNetwork, Serializable):
    def __init__(
            self,
            name_or_scope,
            qfunction,
            action_dim,
            observation_dim,
            learning_rate=1e-3,
            n_update_steps=100,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        super(ArgmaxPolicy, self).__init__(name_or_scope=name_or_scope,
                                           **kwargs)
        self.qfunction = qfunction
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.learning_rate = learning_rate
        self.n_update_steps = n_update_steps

        self.observation_input = tf.placeholder(tf.float32,
                                                [self.observation_dim])
        self.proposed_action = tf.random_uniform([self.action_dim], -1, 1)
        self.loss = - qfunction.output
        self.minimizer_op = tf.train.AdamOptimizer(
            self.learning_rate).minimize(
            self.loss,
            var_list=self.proposed_action)

    def get_action(self, observation):
        assert observation.shape == (self.observation_dim,)
        for _ in range(self.n_update_steps):
            self.sess.run(self.minimizer_op,
                          {self.observation_input: [observation]})
        return self.sess.run(self.proposed_action), {}
