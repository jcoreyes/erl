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
            action_dim=None,
            observation_dim=None,
            observation_input=None,
            reuse=False,
            variable_scope=None,
            **kwargs):
        """

        :param name_or_scope:
        :param output_dim:
        :param env_spec:
        :param action_dim:
        :param observation_dim:
        :param observation_input:
        :param reuse:
        :param variable_scope:
        :param kwargs:
        """
        Serializable.quick_init(self, locals())
        self.output_dim = output_dim
        self.observation_input = observation_input

        assert env_spec or observation_dim
        self.observation_dim = (observation_dim or
                                env_spec.observation_space.flat_dim)

        with tf.variable_scope(name_or_scope, reuse=reuse) as variable_scope:
            if self.observation_input is None:
                self.observation_input = tf.placeholder(
                    tf.float32,
                    [None, self.observation_dim],
                    "_observation")
            super(StateNetwork, self).__init__(
                variable_scope.original_name_scope, **kwargs)
            self._output = self._create_network(self.observation_input)
            self.variable_scope = variable_scope

    def get_weight_tied_copy(self, observation_input=None):
        """
        Return a weight-tied copy of the network. Optionally replace the
        observation input to the network for the returned network.

        :param observation_input: A tensor or placeholder. If not set,
        the observation input to the returned network is the same as this
        network's observation input.
        :return: StateNetwork copy with weights tied to this StateNetwork.
        """
        if observation_input is None:
            observation_input = self.observation_input
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
