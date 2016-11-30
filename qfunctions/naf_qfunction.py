import abc
from qfunctions.nn_qfunction import NNQFunction
from qfunctions.optimizable_q_function import OptimizableQFunction
from qfunctions.separable_q_function import SeparableQFunction


class NAFQFunction(NNQFunction,
                   OptimizableQFunction,
                   SeparableQFunction,
                   metaclass=abc.ABCMeta):
    pass
