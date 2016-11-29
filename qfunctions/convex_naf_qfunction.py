import tensorflow as tf

from core.neuralnet import NeuralNetwork
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
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        self.policy = None
        self.af = None
        self.vf = None
        super(NAFQFunction, self).__init__(
            name_or_scope=name_or_scope,
            **kwargs
        )

    @overrides
    def _create_network(self, observation_input, action_input):
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
        return self.vf.output + self.af.output

    @overrides
    def _generate_inputs(self, action_input, observation_input):
        self.af = ActionConvexQFunction(
            name_or_scope="advantage_function",
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
        )
        return self.af.action_input, self.af.observation_input

    def get_implicit_policy(self):
        if self.policy is None:
            self.policy = ArgmaxPolicy(
                name_or_scope="argmax_policy",
                qfunction=self.af,
            )
        return self.policy

    def get_implicit_value_function(self):
        return self.vf

    def get_implicit_advantage_function(self):
        return self.af


class ArgmaxPolicy(NeuralNetwork, Policy, Serializable):
    """
    A policy that outputs

    pi(s) = argmax_a Q(a, s)

    The policy is optimized using a gradient descent method on the action.
    """

    def __init__(
            self,
            name_or_scope,
            qfunction,
            learning_rate=1e-3,
            n_update_steps=100,
            **kwargs
    ):
        """

        :param name_or_scope:
        :param proposed_action: tf.Variable, which will be optimized for the
        state.
        :param qfunction: Some NNQFunction
        :param action_dim:
        :param observation_dim:
        :param learning_rate: Gradient descent learning rate.
        :param n_update_steps: How many gradient descent steps to take to
        figure out the action.
        :param kwargs:
        """
        Serializable.quick_init(self, locals())
        self.qfunction = qfunction
        self.learning_rate = learning_rate
        self.n_update_steps = n_update_steps

        self.observation_input = qfunction.observation_input
        self.action_dim = qfunction.action_dim
        self.observation_dim = qfunction.observation_dim

        # Normally, this Q function is trained by getting actions. We need
        # to make a copy where the action inputted are generated from
        # internally.
        init = tf.random_uniform([1, self.action_dim],
                                 minval=-1.,
                                 maxval=1.)
        with tf.variable_scope(name_or_scope) as variable_scope:
            super(ArgmaxPolicy, self).__init__(name_or_scope=variable_scope,
                                               **kwargs)
            self.proposed_action = tf.Variable(
                init,
                name="proposed_action")
            self.af_with_proposed_action = self.qfunction.get_weight_tied_copy(
                action_input=self.proposed_action
            )

            self.loss = -self.af_with_proposed_action.output
            self.minimizer_op = tf.train.AdamOptimizer(
                self.learning_rate).minimize(
                self.loss,
                var_list=[self.proposed_action])

    @overrides
    def get_params_internal(self, **tags):
        # Adam optimizer has variables as well, so don't just add
        # `super().get_params_internal()`
        return (self.qfunction.get_params_internal(**tags) +
                tf.get_collection(tf.GraphKeys.VARIABLES, self.scope_name))

    def get_action(self, observation):
        assert observation.shape == (self.observation_dim,)
        for _ in range(self.n_update_steps):
            self.sess.run(self.minimizer_op,
                          {self.observation_input: [observation]})
        return self.sess.run(self.proposed_action), {}
