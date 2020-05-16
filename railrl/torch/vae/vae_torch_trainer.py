from collections import OrderedDict, namedtuple

import numpy as np
import torch
import torch.optim as optim
from railrl.core.loss import LossFunction
from torch import nn as nn

import railrl.torch.pytorch_util as ptu
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.torch.torch_rl_algorithm import TorchTrainer
from railrl.core.logging import add_prefix

VAELosses = namedtuple(
    'VAELoss',
    'vae_loss',
)


class VAETrainer(TorchTrainer, LossFunction):
    def __init__(
            self,
            vae,
            vae_lr=1e-3,
            beta=1,
            optimizer_class=optim.Adam,
    ):
        super().__init__()
        self.vae = vae
        self.beta = beta
        self.vae_optimizer = optimizer_class(
            self.vae.parameters(),
            lr=vae_lr,
        )

    def train_from_torch(self, batch):
        raise NotImplementedError()

    def signal_completed_training_step(self):
        super().signal_completed_training_step()

    def kl_divergence(self, z_mu, logvar):
        return - 0.5 * torch.sum(
            1 + logvar - z_mu.pow(2) - logvar.exp(), dim=1
        ).mean()

    def compute_loss(self, batch, return_statistics=False) -> VAELosses:
        next_obs = batch['raw_next_observations']

        recon, z_mu, z_logvar = self.vae.reconstruct(
            next_obs,
            use_mean=False,
            return_latent_params=True,
        )

        vae_logprob = self.vae.logprob(next_obs, recon)
        recon_loss = -vae_logprob

        kl_divergence = self.kl_divergence(z_mu, z_logvar)
        kl_loss = kl_divergence
        scaled_kl_loss = self.beta * kl_loss
        vae_loss = recon_loss + scaled_kl_loss

        loss = VAELosses(
            vae_loss=vae_loss,
        )
        """
        Save some statistics
        """
        eval_statistics = OrderedDict()
        if return_statistics:
            mean_vae_logprob = vae_logprob.mean()
            mean_kl_divergence = kl_divergence.mean()

            eval_statistics['VAE Log Prob'] = np.mean(ptu.get_numpy(
                mean_vae_logprob
            ))
            eval_statistics['VAE KL'] = np.mean(ptu.get_numpy(
                mean_kl_divergence
            ))
        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.vae,
        ]

    @property
    def optimizers(self):
        return [
            self.vae_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            vae=self.vae,
        )
