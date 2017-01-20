import abc
import tensorflow as tf

from railrl.core.neuralnet import NeuralNetwork
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
            reuse=False,
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
        :param reuse: boolean, reuse variables when creating network?
        :param kwargs: kwargs to be passed to super
        """
        self.setup_serialization(locals())
        self.output_dim = output_dim
        self.reuse = reuse

        assert env_spec or (action_dim and observation_dim)
        if action_dim is None:
            self.observation_dim = env_spec.observation_space.flat_dim
            self.action_dim = env_spec.action_space.flat_dim
        else:
            self.action_dim = action_dim
            self.observation_dim = observation_dim

        with tf.variable_scope(name_or_scope, reuse=reuse) as variable_scope:
            super(StateActionNetwork, self).__init__(variable_scope, **kwargs)
            if action_input is None:
                action_input = tf.placeholder(
                    tf.float32,
                    [None, self.action_dim],
                    "_actions")
            self.action_input = action_input
            if observation_input is None:
                observation_input = tf.placeholder(
                    tf.float32,
                    [None, self.observation_dim],
                    "_observation")
            self.observation_input = observation_input
            self._output = self._create_network(self.observation_input,
                                                self.action_input)
            self.variable_scope = variable_scope

    @property
    @overrides
    def _input_name_to_values(self):
        return dict(
            observation_input=self.observation_input,
            action_input=self.action_input,
        )
