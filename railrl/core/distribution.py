import abc
from typing import Dict
from gym import Space


class DictDistribution(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, batch_size: int, use_env_goal=False):
        pass

    @property
    @abc.abstractmethod
    def spaces(self) -> Dict[str, Space]:
        pass
