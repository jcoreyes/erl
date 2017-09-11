import abc


class Policy(object, metaclass=abc.ABCMeta):
    """
    General policy interface.
    """
    @abc.abstractmethod
    def get_action(self, obs):
        """

        :param obs:
        :return: action, debug_dictionary
        """
        pass

    def reset(self):
        pass


class ExplorationPolicy(Policy):
    def set_num_steps_total(self, t):
        pass


class SerializablePolicy(Policy, metaclass=abc.ABCMeta):
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
