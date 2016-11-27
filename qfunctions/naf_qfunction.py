import abc
from qfunctions.nn_qfunction import NNQFunction
from qfunctions.optimizable_q_function import OptimizableQFunction
from qfunctions.separable_q_function import SeparableQFunction


class NAFQFunction(NNQFunction,
                   OptimizableQFunction,
                   SeparableQFunction):
    @abc.abstractmethod
    def get_implicit_value_function(self):
        return

    @abc.abstractmethod
    def get_implicit_advantage_function(self):
        return

    @abc.abstractmethod
    def _create_network(self, observation_input, action_input):
        return

    @abc.abstractmethod
    def get_implicit_policy(self):
        return