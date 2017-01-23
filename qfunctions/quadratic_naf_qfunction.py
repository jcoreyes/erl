import tensorflow as tf
from typing import Iterable

from railrl.core.neuralnet import NeuralNetwork
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.predictors.mlp_state_network import MlpStateNetwork
from railrl.qfunctions.naf_qfunction import NAFQFunction
from railrl.qfunctions.quadratic_qf import QuadraticQF

from rllab.misc.overrides import overrides


class QuadraticNAF(NAFQFunction):
    @overrides
    def _create_network_internal(self, observation_input, action_input):
        observation_input = self._process_layer(observation_input,
                                                scope_name="observation_input")
        action_input = self._process_layer(action_input,
                                           scope_name="action_input")
        self._vf = MlpStateNetwork(
            name_or_scope="V_function",
            output_dim=1,
            observation_dim=self.observation_dim,
            observation_input=observation_input,
            observation_hidden_sizes=(100, 100),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.identity,
            batch_norm_config=self._batch_norm_config,
        )
        self._policy = FeedForwardPolicy(
            name_or_scope="implicit_policy",
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
            observation_input=observation_input,
            observation_hidden_sizes=(100, 100),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
            batch_norm_config=self._batch_norm_config,
        )
        self._af = QuadraticQF(
            name_or_scope="advantage_function",
            action_input=action_input,
            observation_input=observation_input,
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
            policy=self._policy,
            batch_norm_config=self._batch_norm_config,
        )
        return self._vf.output + self._af.output

    @property
    def _subnetworks(self) -> Iterable[NeuralNetwork]:
        return [self._vf, self._policy, self._af]

    @property
    def implicit_policy(self):
        return self._af.implicit_policy

    @property
    def value_function(self):
        return self._vf

    @property
    def advantage_function(self):
        return self._af

    @property
    def update_weights_ops(self):
        return None
