import abc
from gym import Space


class Distribution(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, batch_size: int):
        pass

    @property
    @abc.abstractmethod
    def space(self) -> Space:
        pass
