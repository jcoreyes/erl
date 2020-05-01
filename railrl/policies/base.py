import abc
from railrl.torch.core import torch_ify, elem_or_tuple_to_numpy

class Policy(object, metaclass=abc.ABCMeta):
    """
    General policy interface.
    """
    @abc.abstractmethod
    def get_action(self, observation):
        """

        :param observation:
        :return: action, debug_dictionary
        """
        pass

    def reset(self):
        pass


class ExplorationPolicy(Policy, metaclass=abc.ABCMeta):
    def set_num_steps_total(self, t):
        pass


class TorchPolicy(ExplorationPolicy):
    def get_action(self, obs_np, ):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs_np, ):
        dist = self.get_dist_from_np(obs_np)
        actions = dist.sample()
        return elem_or_tuple_to_numpy(actions)

    def get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        dist = self(*torch_args, **torch_kwargs)
        return dist
