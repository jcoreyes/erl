import abc
from typing import Iterable

from railrl.core.rl_algorithm import BatchRLAlgorithm
from railrl.core.trainer import Trainer
from railrl.torch.core import PyTorchModule, np_to_pytorch_batch


class TorchBatchRLAlgorithm(BatchRLAlgorithm, metaclass=abc.ABCMeta):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)


class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def train(self, np_batch):
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch)

    @abc.abstractmethod
    def train_from_torch(self, batch):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[PyTorchModule]:
        pass
