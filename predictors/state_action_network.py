import abc

import tensorflow as tf

from railrl.core.neuralnet import NeuralNetwork
from railrl.core.tf_util import create_placeholder
from railrl.misc.rllab_util import get_action_dim, get_observation_dim
from rllab.misc.overrides import overrides


class StateActionNetwork(NeuralNetwork, metaclass=abc.ABCMeta):
    """
    A map from (state, action) to a vector
    """

    def __init__(
            self,
            name_or_scope,
            output_dim,
            env_spec=None,
            action_dim=None,
            observation_dim=None,
            action_input=None,
            observation_input=None,
            **kwargs
    ):
        """
        Create a state-action network.

        :param name_or_scope: a string or VariableScope
        :param output_dim: int, output dimension of this network
        :param env_spec: env spec for an Environment
            :param action_dim: int, action dimension
        :param observation_input: tf.Tensor, observation input. If None,
        a placeholder of shape [None, observation dim] will be made
        :param action_input: tf.Tensor, observation input. If None,
        a placeholder of shape [None, action dim] will be made
        :param kwargs: kwargs to be passed to super
        """
        self.setup_serialization(locals())
        super(StateActionNetwork, self).__init__(name_or_scope, **kwargs)
        self.output_dim = output_dim

        self.action_dim = get_action_dim(
            env_spec=env_spec,
            action_dim=action_dim,
        )
        self.observation_dim = get_observation_dim(
            env_spec=env_spec,
            observation_dim=observation_dim,
        )

        with tf.variable_scope(self.scope_name):
            if action_input is None:
                action_input = create_placeholder(
                    self.action_dim,
                    "action_input",
                )
            if observation_input is None:
                observation_input = create_placeholder(
                    self.observation_dim,
                    "observation_input",
                )
        self.action_input = action_input
        self.observation_input = observation_input
        self._create_network(observation_input=observation_input,
                             action_input=action_input)

    @property
    @overrides
    def _input_name_to_values(self):
        return dict(
            observation_input=self.observation_input,
            action_input=self.action_input,
        )

    # TODO(vpong): make it so that the inputs get automatically processed
