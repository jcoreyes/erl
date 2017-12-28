import abc

from typing import Iterable

from railrl.torch.algos.util import np_to_pytorch_batch
from railrl.torch.core import PyTorchModule
from railrl.torch.algos.rl_algorithm import RLAlgorithm


class TorchRLAlgorithm(RLAlgorithm, metaclass=abc.ABCMeta):
    def get_batch(self, training=True):
        if self.replay_buffer_is_split:
            replay_buffer = self.replay_buffer.get_replay_buffer(training)
        else:
            replay_buffer = self.replay_buffer
        batch = replay_buffer.random_batch(self.batch_size)
        return np_to_pytorch_batch(batch)

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[PyTorchModule]:
        pass

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def cuda(self):
        for net in self.networks:
            net.cuda()
