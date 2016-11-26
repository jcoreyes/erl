import abc
import tensorflow as tf
from core.neuralnet import NeuralNetwork
from rllab.core.serializable import Serializable


class StateActionNetwork(NeuralNetwork):
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
        Serializable.quick_init(self, locals())
        self.output_dim = output_dim
        self.reuse = reuse
        self.action_input = action_input
        self.observation_input = observation_input

        assert env_spec or (action_dim and observation_dim)
        if action_dim is None:
            self.observation_dim = env_spec.observation_space.flat_dim
            self.action_dim = env_spec.action_space.flat_dim
        else:
            self.action_dim = action_dim
            self.observation_dim = observation_dim

        with tf.variable_scope(name_or_scope, reuse=reuse) as variable_scope:
            if self.action_input is None:
                self.action_input = tf.placeholder(
                    tf.float32,
                    [None, self.action_dim],
                    "_actions")
            if self.observation_input is None:
                self.observation_input = tf.placeholder(
                    tf.float32,
                    [None, self.observation_dim],
                    "_observation")
            super(StateActionNetwork, self).__init__(
                variable_scope.original_name_scope, **kwargs)
            self._output = self._create_network(self.observation_input,
                                                self.action_input)
            self.variable_scope = variable_scope

    def get_weight_tied_copy(self, observation_input=None, action_input=None):
        """
        Return a weight-tied copy of the network. Replace the action or
        observation to the network for the returned network.

        :param action_input: A tensorflow Tensor. If not set,
        the action input to the returned network is the same as this network's
        action input.
        :param observation_input: A tensorflow Tensor. If not set,
        the observation input to the returned network is the same as this
        network's observation input.
        :return: StateNetwork copy with weights tied to this StateNetwork.
        """
        assert (observation_input is not None or action_input is not None)
        if observation_input is None:
            observation_input = self.observation_input
        if action_input is None:
            action_input = self.action_input
        return self.get_copy(
            name_or_scope=self.variable_scope,
            observation_input=observation_input,
            action_input=action_input,
            reuse=True,
        )

    @abc.abstractmethod
    def _create_network(self, observation_input, action_input):
        """
        Create a network whose inputs are observation_input and action_input.

        :param action_input: A tensor/placeholder.
        :param observation_input: A tensor/placeholder.
        :return: A tensor.
        """
        return
