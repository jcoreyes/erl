import abc

import tensorflow as tf

from railrl.core.neuralnet import NeuralNetwork
from railrl.core.tf_util import create_placeholder
from railrl.misc.rllab_util import get_observation_dim
from rllab.misc.overrides import overrides


class StateNetwork(NeuralNetwork, metaclass=abc.ABCMeta):
    """
    A map from state to a vector
    """

    def __init__(
            self,
            name_or_scope,
            output_dim,
            env_spec=None,
            observation_dim=None,
            observation_input=None,
            **kwargs):
        """
        Create a state network.

        :param name_or_scope: a string or VariableScope
        :param output_dim: int, output dimension of this network
        :param env_spec: env spec for an Environment
        :param action_dim: int, action dimension
        :param observation_dim: int, observation dimension
        :param observation_input: tf.Tensor, observation input. If None,
        a placeholder of shape [None, observation dim] will be made
        :param reuse: boolean, reuse variables when creating network?
        :param kwargs: kwargs to be passed to super
        """
        self.setup_serialization(locals())
        super(StateNetwork, self).__init__(name_or_scope, **kwargs)
        self.output_dim = output_dim

        self.observation_dim = get_observation_dim(
            env_spec=env_spec,
            observation_dim=observation_dim,
        )

        with tf.variable_scope(self.scope_name):
            if observation_input is None:
                observation_input = create_placeholder(
                    self.observation_dim,
                    "observation_input",
                )
        self.observation_input = observation_input
        self._create_network(observation_input=observation_input)

    @property
    @overrides
    def _input_name_to_values(self):
        return dict(
            observation_input=self.observation_input,
        )

