import abc


class LossFunction(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute_loss(self, batch, **kwargs):
        pass
