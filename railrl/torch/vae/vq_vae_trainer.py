from collections import OrderedDict
import os
from os import path as osp
import numpy as np
import torch
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

class VQ_VAETrainer(ConvVAETrainer):

    def train_batch(self, epoch, batch):
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.compute_loss(epoch, batch, False)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test_batch(
            self,
            epoch,
            batch,
    ):
        self.model.eval()
        loss = self.compute_loss(epoch, batch, True)

    def encode_dataset(self, dataset, epoch):
        encoding_list = []
        save_dir = osp.join(self.log_dir, 'dataset_latents_%d.npy' % epoch)
        for i in range(len(dataset)):
            obs = dataset.random_batch(self.batch_size)["x_t"]
            encodings = self.model.encode(obs)
            encoding_list.append(encodings)
        encodings = ptu.get_numpy(torch.cat(encoding_list))
        np.save(save_dir, encodings)

    def train_epoch(self, epoch, dataset, batches=100):
        if epoch % 100 == 0 and epoch > 0:
            self.encode_dataset(dataset, epoch)

        start_time = time.time()
        for b in range(batches):
            self.train_batch(epoch, dataset.random_batch(self.batch_size))
        self.eval_statistics["train/epoch_duration"].append(time.time() - start_time)

    def test_epoch(self, epoch, dataset, batches=10):
        start_time = time.time()
        for b in range(batches):
            self.test_batch(epoch, dataset.random_batch(self.batch_size))
        self.eval_statistics["test/epoch_duration"].append(time.time() - start_time)

    def compute_loss(self, epoch, batch, test=False):
        prefix = "test/" if test else "train/"
        beta = float(self.beta_schedule.get_value(epoch))
        obs = batch["x_t"]
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
        # self.model.eval()
        # batch, _ = self.eval_data["test/last_batch"]
        # sample = self.model.sample_prior(64)
        # sample = self.model.decode(sample, batch[self.key_to_reconstruct])[0].cpu()
        # save_dir = osp.join(self.log_dir, 's%d.png' % epoch)
        # save_image(
        #     sample.data.view(64, 3, self.imsize, self.imsize).transpose(2, 3),
        #     save_dir
        # )

        # x0 = batch["x_0"]
        # x0_img = x0[:64].narrow(start=0, length=self.imlength // 2, dim=1).contiguous().view(
        #     -1,
        #     3,
        #     self.imsize,
        #     self.imsize
        # ).transpose(2, 3)
        # save_dir = osp.join(self.log_dir, 'x0_%d.png' % epoch)
        # save_image(x0_img.data.cpu(), save_dir)
