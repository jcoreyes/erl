import torch

from railrl.torch.torch_rl_algorithm import TorchTrainer
from railrl.torch.core import np_to_pytorch_batch

class HERTrainer(TorchTrainer):
    def __init__(self, base_trainer: TorchTrainer):
        super().__init__()
        self._base_trainer = base_trainer

    def train_from_torch(self, batch):
        obs = batch['observations']
        next_obs = batch['next_observations']
        goals = batch['resampled_goals']
        batch['observations'] = torch.cat((obs, goals), dim=1)
        batch['next_observations'] = torch.cat((next_obs, goals), dim=1)
        self._base_trainer.train_from_torch(batch)

    def get_diagnostics(self):
        return self._base_trainer.get_diagnostics()

    def end_epoch(self, epoch):
        self._base_trainer.end_epoch(epoch)

    @property
    def networks(self):
        return self._base_trainer.networks

    def get_snapshot(self):
        return self._base_trainer.get_snapshot()

    def pretrain_policy_with_bc(self):
        self._base_trainer.get_batch_from_buffer = self.get_batch_from_buffer
        return self._base_trainer.pretrain_policy_with_bc()

    def get_batch_from_buffer(self, replay_buffer, batch_size):
        batch = replay_buffer.random_batch(batch_size)
        batch = np_to_pytorch_batch(batch)
        obs = batch['observations']
        next_obs = batch['next_observations']
        goals = batch['resampled_goals']
        batch['observations'] = torch.cat((obs, goals), dim=1)
        batch['next_observations'] = torch.cat((next_obs, goals), dim=1)
        return batch
