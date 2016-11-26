import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.core.parameterized import Parameterized


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

    @overrides
    def get_params_internal(self, **tags):
        # TODO(vpong): This is a big hack! Fix this
        if 'regularizable' in tags:
            return [v
                    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               self.scope_name)
                    if 'bias' not in v.name]
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 self.scope_name)

    def get_copy(self, **kwargs):
        return Serializable.clone(
            self,
            **kwargs
        )


