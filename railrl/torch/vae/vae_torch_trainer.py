from collections import OrderedDict, namedtuple
import os.path as osp
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

from railrl.core import logger
from railrl.core.logging import add_prefix
from railrl.core.loss import LossFunction
from railrl.misc.eval_util import create_stats_ordered_dict
import railrl.torch.pytorch_util as ptu
from railrl.torch.torch_rl_algorithm import TorchTrainer
from railrl.core.logging import add_prefix
from railrl.core.timer import timer
from railrl.visualization.image import combine_images_into_grid

VAELosses = namedtuple(
    'VAELoss',
    'vae_loss',
)
LossStatistics = OrderedDict


class VAETrainer(TorchTrainer, LossFunction):
    def __init__(
            self,
            vae,
            vae_lr=1e-3,
            beta=1,
            loss_scale=1.0,
            vae_visualization_config=None,
            optimizer_class=optim.Adam,
    ):
        super().__init__()
        self.vae = vae
        self.beta = beta
        self.vae_optimizer = optimizer_class(
            self.vae.parameters(),
            lr=vae_lr,
        )
        self._need_to_update_eval_statistics = True
        self.loss_scale = loss_scale
        self.eval_statistics = OrderedDict()

        self.vae_visualization_config = vae_visualization_config
        if not self.vae_visualization_config:
            self.vae_visualization_config = {}

    def train_from_torch(self, batch):
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        self.vae_optimizer.zero_grad()
        losses.vae_loss.backward()
        self.vae_optimizer.step()

        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            self._need_to_update_eval_statistics = False
            self.example_obs_batch = batch['raw_next_observations']

    def kl_divergence(self, z_mu, logvar):
        return - 0.5 * torch.sum(
            1 + logvar - z_mu.pow(2) - logvar.exp(), dim=1
        ).mean()

    def compute_loss(
        self,
        batch,
        skip_statistics=False
    ) -> Tuple[VAELosses, LossStatistics]:
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
            vae_loss=vae_loss * self.loss_scale,
        )
        """
        Save some statistics
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            mean_vae_logprob = vae_logprob.mean()
            mean_kl_divergence = kl_divergence.mean()

            eval_statistics['Log Prob'] = np.mean(ptu.get_numpy(
                mean_vae_logprob
            ))
            eval_statistics['KL'] = np.mean(ptu.get_numpy(
                mean_kl_divergence
            ))
        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        self.dump_debug_images(epoch, **self.vae_visualization_config)

    def dump_debug_images(
        self,
        epoch,
        dump_images=True,
        num_recons=10,
        num_samples=25
    ):
        if not dump_images:
            return
        example_obs_batch_np = ptu.get_numpy(self.example_obs_batch)
        recon_examples_np = ptu.get_numpy(
            self.vae.reconstruct(self.example_obs_batch, use_mean=True))

        top_row_example = example_obs_batch_np[:num_recons]
        bottom_row_recon = np.clip(recon_examples_np, 0, 1)[:num_recons]

        recon_vis = combine_images_into_grid(
            imgs=list(top_row_example) + list(bottom_row_recon),
            imwidth=example_obs_batch_np.shape[2],
            imheight=example_obs_batch_np.shape[3],
            max_num_cols=len(top_row_example),
            image_format='CWH',
        )

        logdir = logger.get_snapshot_dir()
        cv2.imwrite(
            osp.join(logdir, '{}_recons.png'.format(epoch)),
            recon_vis,
        )

        vae_samples = np.clip(self.vae.sample_np(num_samples), 0, 1)
        vae_sample_vis = combine_images_into_grid(
            imgs=list(vae_samples),
            imwidth=example_obs_batch_np.shape[2],
            imheight=example_obs_batch_np.shape[3],
            image_format='CWH',
        )
        cv2.imwrite(
            osp.join(logdir, '{}_vae_samples.png'.format(epoch)),
            vae_sample_vis,
        )

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
