import abc
import tensorflow as tf

from railrl.core import tf_util
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.core.parameterized import Parameterized

ALLOWABLE_TAGS = ['regularizable']


def negate(function):
    return lambda x: not function(x)


class NeuralNetwork(Parameterized, Serializable):
    """
    Any neural network.
    """
    def __init__(self, name_or_scope, **kwargs):
        super().__init__(**kwargs)
        Serializable.quick_init(self, locals())
        if type(name_or_scope) is str:
            self.scope_name = name_or_scope
        else:
            self.scope_name = name_or_scope.original_name_scope
        self._output = None
        self._sess = None

    @property
    def sess(self):
        if self._sess is None:
            self._sess = tf.get_default_session()
        return self._sess

    @sess.setter
    def sess(self, value):
        self._sess = value

    @property
    def output(self):
        """
        :return: Tensor/placeholder/op. Output of this network.
        """
        return self._output

    @property
    def training_output(self):
        """
        :return: Tensor/placeholder/op. Training output of this network.
        """
        return self.output

    def process_layer(self, previous_layer):
        """
        This should be done called between every layer, i.e.

        a = self.process_layer(linear(x))
        b = self.process_layer(linear(relu(a)))

        :param previous_layer:
        :return:
        """
        return previous_layer

    @overrides
    def get_params_internal(self, **tags):
        for key in tags.keys():
            if key not in ALLOWABLE_TAGS:
                raise KeyError(
                    "Tag not allowed: {0}. Allowable tags: {1}".format(
                        key,
                        ALLOWABLE_TAGS))
        # TODO(vpong): This is a big hack! Fix this
        filters = []
        if 'regularizable' in tags:
            regularizable_vars = tf_util.get_regularizable_variables(
                self.scope_name)
            if tags['regularizable']:
                reg_filter = lambda v: v in regularizable_vars
            else:
                reg_filter = lambda v: v not in regularizable_vars
            filters.append(reg_filter)

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      self.scope_name)
        return list(filter(lambda v: all(f(v) for f in filters), variables))

    def get_copy(self, **kwargs):
        return Serializable.clone(
            self,
            **kwargs
        )

    def get_weight_tied_copy(self, **inputs):
        """
        Return a weight-tied copy of the network. Replace the action or
        observation to the network for the returned network.

        :param inputs: Dictionary, of the form
        {
            'input_x': self.input_x,
            'input_y': self.input_y,
        }

        :return: StateNetwork copy with weights tied to this StateNetwork.
        """
        assert len(inputs) > 0
        for input_name, input_value in self._input_name_to_values.items():
            if input_name not in inputs:
                inputs[input_name] = input_value
        return self.get_copy(
            name_or_scope=self.variable_scope,
            reuse=True,
            **inputs
        )

    @property
    @abc.abstractmethod
    def _input_name_to_values(self):
        pass

    @abc.abstractmethod
    def _create_network(self, **inputs):
        """
        Create a network whose inputs are given.

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
