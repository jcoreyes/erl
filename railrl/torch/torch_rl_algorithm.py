import abc
from typing import Iterable

from railrl.core.rl_algorithm import RLAlgorithm, BatchRlAlgorithm
from railrl.core.trainer import Trainer
from railrl.torch.core import PyTorchModule
from railrl.torch import pytorch_util as ptu


class TorchRLAlgorithm(RLAlgorithm, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[PyTorchModule]:
        pass

    def to(self, device=None):
        if device is None:
            device = ptu.device
        for net in self.networks:
            net.to(device)


class TorchBatchRLAlgorithm(BatchRlAlgorithm, metaclass=abc.ABCMeta):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)


class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[PyTorchModule]:
        pass
