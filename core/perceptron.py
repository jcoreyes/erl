import tensorflow as tf

from rllab.core.serializable import Serializable
from core.tf_util import linear, BIAS_DEFAULT_NAME, WEIGHT_DEFAULT_NAME
from core.neuralnet import NeuralNetwork


class Perceptron(NeuralNetwork):
    """A perceptron, where output = W * input + b"""
    def __init__(
            self,
            name_or_scope,
            input_tensor,
            input_size,
            output_size,
            W_name=WEIGHT_DEFAULT_NAME,
            b_name=BIAS_DEFAULT_NAME,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        super().__init__(name_or_scope, **kwargs)

        with tf.variable_scope(name_or_scope) as variable_scope:
            self._output = linear(
                input_tensor,
                input_size,
                output_size,
                W_name=W_name,
                b_name=b_name,
            )
