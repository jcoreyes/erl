from collections import (
    defaultdict,
    OrderedDict,
)
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
        trainer_loss_scales: MutableMapping[TorchTrainer, float]=None,

    ):
        super().__init__()
        if len(trainers) == 0:
            raise ValueError("Need at least one trainer")
        self._trainers = trainers
        self._optimizers = optimizers
        self._trainer_stats = {}

        for name, trainer in self._trainers.items():
            if hasattr(trainer, 'optimizers'):
                trainer_optimizers = trainer.optimizers
                for optimizer in trainer_optimizers:
                    assert optimizer in self._optimizers, (
                        'Joint loss trainer {} missing optimizer'.format(name))

        self.trainer_loss_scales = trainer_loss_scales
        if self.trainer_loss_scales is None:
            self.trainer_loss_scales = defaultdict(lambda trainer: 1)
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        # Compute losses
        trainer_losses = []
        for trainer in self._trainers.values():
            trainer_loss, eval_stats = trainer.compute_loss(
                batch,
                self._need_to_update_eval_statistics
            )
            trainer_losses.append(trainer_loss)

            if self._need_to_update_eval_statistics:
                self._trainer_stats[trainer] = eval_stats

        # Clear optimizer gradients
        for optimizer in self._optimizers:
            optimizer.zero_grad()

        # Compute gradients
        for trainer_loss in trainer_losses:
            trainer_loss_scale = self.trainer_loss_scales[trainer]
            for loss in trainer_loss:
                (trainer_loss_scale * loss).backward()

        # Backprop gradients
        for optimizer in self._optimizers:
            optimizer.step()

        # Cleanup
        for trainer in self._trainers.values():
            trainer.signal_completed_training_step()
        self.signal_completed_training_step()

    def signal_completed_training_step(self):
        super().signal_completed_training_step()
        if self._need_to_update_eval_statistics:
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False

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
        self._need_to_update_eval_statistics = True

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
        stats = OrderedDict()
        for trainer_name, trainer in self._trainers.items():
            for k, v in self._trainer_stats[trainer].items():
                if trainer_name:
                    new_k = '{}/{}'.format(trainer_name, k)
                    stats[new_k] = v
                else:
                    stats[k] = v
        return stats
