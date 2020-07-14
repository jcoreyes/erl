from collections import OrderedDict
import os
from os import path as osp
import numpy as np
import torch
import argparse
import json
import logging
import pickle
from railrl.core.loss import LossFunction
from railrl.torch.vae.vae_trainer import ConvVAETrainer
from torch import optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.utils import save_image
from railrl.data_management.images import normalize_image
from railrl.core import logger
import railrl.core.util as util
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.misc.ml_util import ConstantSchedule
from railrl.torch import pytorch_util as ptu
from railrl.torch.data import (
    ImageDataset, InfiniteWeightedRandomSampler,
    InfiniteRandomSampler,
)
from railrl.torch.core import np_to_pytorch_batch
import collections
import time

#from biva.datasets import get_binmnist_datasets, get_cifar10_datasets
from railrl.torch.vae.biva_pytorch.biva.evaluation import VariationalInference
from railrl.torch.vae.biva_pytorch.biva.model import \
    DeepVae, get_deep_vae_mnist, get_deep_vae_cifar, VaeStage, LvaeStage, BivaStage
from railrl.torch.vae.biva_pytorch.biva.utils import \
    LowerBoundedExponentialLR, training_step, test_step, summary2logger, save_model, load_model, \
    sample_model, DiscretizedMixtureLogits
#from booster.utils import EMA
from torch.distributions import Bernoulli


class BIVATrainer(ConvVAETrainer, LossFunction):
    def __init__(
            self,
            model,
            batch_size=128,
            log_interval=0,
            beta=1.0,
            beta_schedule=None,
            lr=2e-3,
            do_scatterplot=False,
            normalize=False,
            mse_weight=0.1,
            is_auto_encoder=False,
            background_subtract=False,
            linearity_weight=0.0,
            distance_weight=0.0,
            loss_weights=None,
            use_linear_dynamics=False,
            use_parallel_dataloading=False,
            train_data_workers=2,
            skew_dataset=False,
            skew_config=None,
            priority_function_kwargs=None,
            start_skew_epoch=0,
            weight_decay=0,
            key_to_reconstruct='x_t',
            num_epochs=500,

            ema=0.9995,
            nr_mix=10,
            free_bits=2.0,
            
        ):
        super().__init__(
            model,
            batch_size,
            log_interval,
            beta,
            beta_schedule,
            lr,
            do_scatterplot,
            normalize,
            mse_weight,
            is_auto_encoder,
            background_subtract,
            linearity_weight,
            distance_weight,
            loss_weights,
            use_linear_dynamics,
            use_parallel_dataloading,
            train_data_workers,
            skew_dataset,
            skew_config,
            priority_function_kwargs,
            start_skew_epoch,
            weight_decay,
            key_to_reconstruct,
            num_epochs
        )
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=lr, betas=(0.9, 0.999,))
        self.scheduler = LowerBoundedExponentialLR(self.optimizer, 0.999999, 0.0001)
        self.likelihood = DiscretizedMixtureLogits(nr_mix)
        #self.likelihood = Bernoulli
        self.evaluator = VariationalInference(self.likelihood, iw_samples=1)
        
        #IF ABOVE DOESNT WORK, TRY OTHER LIKELIHOOD
        #self.ema = EMA(model, ema)

        n_latents = len(self.model.latents)
        n_latents = 2 * n_latents - 1
        freebits = [free_bits] * n_latents
        self.kwargs = {'beta': beta, 'freebits': freebits}

    def train_batch(self, epoch, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch, epoch, False)

        loss.backward()
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
        #self.ema.update()


    def compute_loss(self, batch, epoch=-1, test=False):
        prefix = "test/" if test else "train/"
        self.kwargs['beta'] = float(self.beta_schedule.get_value(epoch))
        #model = self.ema.model if test else self.model
        obs = batch[self.key_to_reconstruct]

        loss, diagnostics, output = self.evaluator(self.model, obs, **self.kwargs)
        recon = output['x_'].reshape(obs.shape[0], -1).detach()
        #recon = self.likelihood(logits=recon).sample().reshape(obs.shape[0], -1).detach()

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics["beta"] = self.kwargs['beta']
        self.eval_statistics[prefix + "kles"].append(diagnostics['loss']['kl'].mean().item())
        self.eval_statistics[prefix + "log_prob"].append(diagnostics['loss']['nll'].mean().item())
        self.eval_statistics[prefix + "elbo"].append(diagnostics['loss']['elbo'].mean().item())
        self.eval_statistics[prefix + "bpd"].append(diagnostics['loss']['bpd'].mean().item())
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_data[prefix + "last_batch"] = (obs, recon)

        return loss

    def dump_samples(self, epoch):
        save_dir = osp.join(self.log_dir, 'samples_%d.png' % epoch)
        n_samples = 64
        
        samples = self.model.sample_from_prior(n_samples).get('x_')
        #samples = self.likelihood(logits=samples).sample()
        save_image(
            samples.data.view(n_samples, self.input_channels, self.imsize, self.imsize).transpose(2, 3),
            save_dir
        )