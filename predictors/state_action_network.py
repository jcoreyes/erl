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

    def setup_serialization(self, init_locals):
        # TODO(vpong): fix this
        # Serializable.quick_init_for_clone(self, init_locals)
        # init_locals_copy = dict(init_locals.items())
        # if 'kwargs' in init_locals:
        #     init_locals_copy['kwargs'].pop('action_input', None)
        #     init_locals_copy['kwargs'].pop('observation_input', None)
        # Serializable.quick_init(self, init_locals_copy)
        Serializable.quick_init(self, init_locals)
