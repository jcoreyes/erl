import abc


class SerializablePolicy(object, metaclass=abc.ABCMeta):
    """
    Policy that can be serialized.
    """
    def get_param_values(self):
        return None

    def set_param_values(self, values):
        pass

    @abc.abstractmethod
    def get_action(self, obs):
        pass

    def reset(self):
        pass
