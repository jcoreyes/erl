from collections import OrderedDict
import os
from os import path as osp
import numpy as np
import torch
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


class VQ_VAETrainer(ConvVAETrainer, LossFunction):

    def train_batch(self, epoch, batch):
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.compute_loss(batch, epoch, False)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test_batch(
            self,
            epoch,
            batch,
    ):
        self.model.eval()
        loss = self.compute_loss(batch, epoch, True)

    def encode_dataset(self, dataset):
        encoding_list = []
        save_dir = osp.join(self.log_dir, 'dataset_latents.npy')
        for i in range(len(dataset)):
            obs = dataset.random_batch(self.batch_size)["x_t"]
            encodings = self.model.encode(obs, cont=False)
            encoding_list.append(encodings)
        encodings = ptu.get_numpy(torch.cat(encoding_list))
        np.save(save_dir, encodings)

    def train_epoch(self, epoch, dataset, batches=100):
        if epoch % 50 == 0 and epoch > 0:
           self.encode_dataset(dataset)

        start_time = time.time()
        for b in range(batches):
            self.train_batch(epoch, dataset.random_batch(self.batch_size))
        self.eval_statistics["train/epoch_duration"].append(time.time() - start_time)

    def test_epoch(self, epoch, dataset, batches=10):
        start_time = time.time()
        for b in range(batches):
            self.test_batch(epoch, dataset.random_batch(self.batch_size))
        self.eval_statistics["test/epoch_duration"].append(time.time() - start_time)

    def compute_loss(self, batch, epoch=-1, test=False):
        prefix = "test/" if test else "train/"
        beta = float(self.beta_schedule.get_value(epoch))
        obs = batch[self.key_to_reconstruct]
        vq_loss, quantized, data_recon, perplexity, recon_error = self.model.compute_loss(obs)
        loss = vq_loss + recon_error

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_statistics[prefix + "Recon Error"].append(recon_error.item())
        self.eval_statistics[prefix + "VQ Loss"].append(vq_loss.item())
        self.eval_statistics[prefix + "Perplexity"].append(perplexity.item())

        self.eval_data[prefix + "last_batch"] = (obs, data_recon)

        return loss

    def dump_samples(self, epoch):
        return

class CVQVAETrainer(VQ_VAETrainer):

    def encode_dataset(self, dataset):
        encoding_list = []
        save_dir = osp.join(self.log_dir, 'dataset_latents.npy')
        for i in range(len(dataset)):
            batch = dataset.random_batch(self.batch_size)
            encodings = self.model.encode(batch["x_t"], batch["env"], cont=False)
            encoding_list.append(encodings)
        encodings = ptu.get_numpy(torch.cat(encoding_list))
        np.save(save_dir, encodings)

    def test_epoch(self, epoch, dataset, batches=10):
        start_time = time.time()
        for b in range(batches):
            self.test_batch(epoch, dataset.random_batch(self.batch_size))
        self.eval_statistics["test/epoch_duration"].append(time.time() - start_time)

    def compute_loss(self, batch, epoch=-1, test=False):
        prefix = "test/" if test else "train/"
        beta = float(self.beta_schedule.get_value(epoch))
        obs, cond = batch["x_t"], batch["env"]
        vq_losses, perplexities, recons, errors = self.model.compute_loss(obs, cond)
        loss = sum(vq_losses) + sum(errors)

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_statistics[prefix + "Obs Recon Error"].append(errors[0].item())
        self.eval_statistics[prefix + "Cond Obs Recon Error"].append(errors[1].item())
        self.eval_statistics[prefix + "VQ Loss"].append(vq_losses[0].item())
        self.eval_statistics[prefix + "Perplexity"].append(perplexities[0].item())
        self.eval_statistics[prefix + "Cond VQ Loss"].append(vq_losses[1].item())
        self.eval_statistics[prefix + "Cond Perplexity"].append(perplexities[1].item())
        self.eval_data[prefix + "last_batch"] = (batch, recons[0], recons[1])

        return loss

    def dump_reconstructions(self, epoch):
        batch, reconstructions, env_reconstructions = self.eval_data["test/last_batch"]
        obs = batch["x_t"]
        env = batch["env"]
        n = min(obs.size(0), 8)
        comparison = torch.cat([
            env[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3),
            obs[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3),
            reconstructions.view(
                self.batch_size,
                3,
                self.imsize,
                self.imsize,
            )[:n].transpose(2, 3),
            env_reconstructions.view(
                self.batch_size,
                3,
                self.imsize,
                self.imsize,
            )[:n].transpose(2, 3)
        ])
        save_dir = osp.join(self.log_dir, 'r%d.png' % epoch)
        save_image(comparison.data.cpu(), save_dir, nrow=n)

    def dump_samples(self, epoch):
        return
