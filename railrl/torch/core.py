import abc

from torch import nn as nn

from railrl.torch.pytorch_util import copy_model_params
from rllab.core.serializable import Serializable


class PyTorchModule(nn.Module, Serializable, metaclass=abc.ABCMeta):

    def get_param_values(self):
        return [param.data for param in self.parameters()]

    def set_param_values(self, param_values):
        for param, value in zip(self.parameters(), param_values):
            param.data = value

    def copy(self):
        copy = Serializable.clone(self)
        copy_model_params(self, copy)
        return copy

    def save_init_params(self, locals):
        """
        Should call this FIRST THING in the __init__ method if you ever want
        to serialize or clone this network.
        
        Usage:
        ```
        def __init__(self, ...):
            self.init_serialization(locals())
            ...
        ```
        :param locals: 
        :return: 
        """
        Serializable.quick_init(self, locals)