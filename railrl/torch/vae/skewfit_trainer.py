from collections import OrderedDict
import os
from os import path as osp
import numpy as np
import torch
from torch import optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
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

class SkewFitTrainer(object):
    def __init__(
            self,
            trainer,
            replay_buffer,
            method,
            power,
            skew_batch_size=512,
    ):
        self.trainer = trainer
        self.replay_buffer = replay_buffer
        self.method = method
        self.power = power
        self.skew_batch_size = skew_batch_size

    def update_train_weights(self):
        self._train_weights = self._compute_train_weights()

    def _compute_train_weights(self):
        batch_size = self.skew_batch_size
        size = self.train_dataset.shape[0]
        next_idx = min(batch_size, size)
        cur_idx = 0
        weights = np.zeros(size)
        while cur_idx < self.train_dataset.shape[0]:
            idxs = np.arange(cur_idx, next_idx)
            data = self.train_dataset[idxs, :]
            if self.method == 'squared_error':
                weights[idxs] = self._reconstruction_squared_error_np_to_np(
                    data,
                    **self.priority_function_kwargs
                ) ** self.power
            elif self.method == 'kl':
                weights[idxs] = self._kl_np_to_np(data, **self.priority_function_kwargs)
            elif self.method == 'vae_prob':
                data = normalize_image(data)
                weights[idxs] = compute_p_x_np_to_np(self.model, data, power=self.power, **self.priority_function_kwargs)
            elif self.method == 'inv_exp_elbo':
                data = normalize_image(data)
                weights[idxs] = inv_exp_elbo(self.model, data, beta=self.beta) ** power
            else:
                raise NotImplementedError('Method {} not supported'.format(self.method))
            cur_idx = next_idx
            next_idx += batch_size
            next_idx = min(next_idx, size)

        if method == 'vae_prob':
            weights = relative_probs_from_log_probs(weights)
        return weights
