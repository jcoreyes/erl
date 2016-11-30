import abc
import tensorflow as tf
from core.neuralnet import NeuralNetwork
from rllab.core.serializable import Serializable


class StateNetwork(NeuralNetwork):
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
            reuse=False,
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
        self.output_dim = output_dim
        self.observation_input = observation_input

        assert env_spec or observation_dim
        self.observation_dim = (observation_dim or
                                env_spec.observation_space.flat_dim)

        with tf.variable_scope(name_or_scope, reuse=reuse) as variable_scope:
            self.observation_input = self._generate_observation_input(
                observation_input
            )
            super(StateNetwork, self).__init__(variable_scope, **kwargs)
            self._output = self._create_network(self.observation_input)
            self.variable_scope = variable_scope

    def _generate_observation_input(self, observation_input):
        if observation_input is None:
            observation_input = tf.placeholder(
                tf.float32,
                [None, self.observation_dim],
                "_observation")
        return observation_input

    def get_weight_tied_copy(self, observation_input):
        """
        Return a weight-tied copy of the network, with the observation input
        replaced.

        :param observation_input: A tensorflow Tensor. If not set,
        the observation input to the returned network is the same as this
        network's observation input.
        :return: StateNetwork copy with weights tied to this StateNetwork.
        """
        return self.get_copy(
            name_or_scope=self.variable_scope,
            observation_input=observation_input,
            reuse=True,
        )

    @abc.abstractmethod
    def _create_network(self, observation_input):
        """
        Create a network whose input is observation_input.

        :param observation_input: A tensor/placeholder.
        :return: A tensor.
        """
        return

    def setup_serialization(self, init_locals):
        # TODO(vpong): fix this
        # Serializable.quick_init_for_clone(self, init_locals)
        # init_locals_copy = dict(init_locals.items())
        # if 'kwargs' in init_locals:
        #     init_locals_copy['kwargs'].pop('observation_input', None)
        # Serializable.quick_init(self, init_locals_copy)
        Serializable.quick_init(self, init_locals)
