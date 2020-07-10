import abc
from typing import Dict
from gym import Space


class DictDistribution(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, batch_size: int):
        pass

    @abc.abstractmethod
    def get_sample_mode(self) -> str:
        pass

    @abc.abstractmethod
    def set_sample_mode(self, mode: str):
        pass

    @property
    @abc.abstractmethod
    def spaces(self) -> Dict[str, Space]:
        pass
