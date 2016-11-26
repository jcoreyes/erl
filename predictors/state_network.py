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
            scope_name,
            output_dim,
            env_spec=None,
            action_dim=None,
            observation_dim=None,
            observation_input=None,
            reuse=False,
            **kwargs):
        Serializable.quick_init(self, locals())
        self.output_dim = output_dim
        self.observation_input = observation_input

        assert env_spec or observation_dim
        self.observation_dim = (observation_dim or
                                env_spec.observation_space.flat_dim)

        with tf.variable_scope(scope_name, reuse=reuse) as variable_scope:
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
        return self.get_copy(
            scope_name=self.scope_name,
            observation_input=observation_input,
            reuse=True,
        )

    @property
    def output(self):
        return self._output

    @abc.abstractmethod
    def _create_network(self, observation_input):
        return
