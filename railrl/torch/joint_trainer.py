from collections import OrderedDict
from typing import (
    List,
    MutableMapping,
)

from torch.optim import Optimizer

from railrl.torch.torch_rl_algorithm import TorchTrainer

OrderedDictType = MutableMapping


class JointLossTrainer(TorchTrainer):
    """
    Combine multiple trainers by their loss functions.
    Usage:
    ```
    trainer1 = ...
    trainer2 = ...
    trainers = OrderedDict([
        ('sac', sac_trainer),
        ('vae', vae_trainer),
    ])
    joint_trainer = JointTrainer(
        trainers,
        optimizers=[qf_optimizer, vae_optimizer, ...]
    )
    algorithm = RLAlgorithm(trainer=joint_trainer, ...)
    algorithm.train()
    ```
    And then in the logs, the output will be of the fomm:
    ```
    trainer/sac/...
    trainer/vae/...
    ```
    """
    def __init__(
        self,
        trainers: OrderedDictType[str, TorchTrainer],
        optimizers: List[Optimizer],
    ):
        super().__init__()
        if len(trainers) == 0:
            raise ValueError("Need at least one trainer")
        self._trainers = trainers
        self._optimizers = optimizers

        for name, trainer in self._trainers.items():
            if hasattr(trainer, 'optimizers'):
                trainer_optimizers = trainer.optimizers
                for optimizer in trainer_optimizers:
                    assert optimizer in self._optimizers, (
                        'Joint loss trainer {} missing optimizer'.format(name))

    def train_from_torch(self, batch):
        # Compute losses
        trainer_losses = []
        for trainer in self._trainers.values():
            trainer_losses.append(trainer.compute_loss(batch))

        # Clear optimizer gradients
        for optimizer in self._optimizers:
            optimizer.zero_grad()

        # Compute gradients
        for trainer_loss in trainer_losses:
            for loss in trainer_loss:
                loss.backward()

        # Backprop gradients
        for optimizer in self._optimizers:
            optimizer.step()

        # Cleanup
        for trainer in self._trainers.values():
            trainer.signal_completed_training_step()

    @property
    def networks(self):
        for trainer in self._trainers.values():
            for net in trainer.networks:
                yield net

    @property
    def optimizers(self):
        return self._optimizers

    def end_epoch(self, epoch):
        for trainer in self._trainers.values():
            trainer.end_epoch(epoch)

    def get_snapshot(self):
        snapshot = {}
        for trainer_name, trainer in self._trainers.items():
            for k, v in trainer.get_snapshot().items():
                if trainer_name:
                    new_k = '{}/{}'.format(trainer_name, k)
                    snapshot[new_k] = v
                else:
                    snapshot[k] = v
        return snapshot

    def get_diagnostics(self):
        stats = {}
        for trainer_name, trainer in self._trainers.items():
            for k, v in trainer.get_diagnostics().items():
                if trainer_name:
                    new_k = '{}/{}'.format(trainer_name, k)
                    stats[new_k] = v
                else:
                    stats[k] = v
        return stats
