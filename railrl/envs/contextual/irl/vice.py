import warnings
from typing import Any, Callable, Dict, List

import numpy as np
from gym.spaces import Box, Dict
from multiworld.core.multitask_env import MultitaskEnv

from railrl import pythonplusplus as ppp
from railrl.core.distribution import DictDistribution
from railrl.envs.contextual import ContextualRewardFn
from railrl.envs.contextual.contextual_env import (
    ContextualDiagnosticsFn,
    Path,
    Context,
    Diagnostics,
)
from railrl.envs.images import Renderer
from railrl.core.loss import LossFunction
from railrl.torch import pytorch_util as ptu

Observation = Dict
Goal = Any
GoalConditionedDiagnosticsFn = Callable[
    [List[Path], List[Goal]],
    Diagnostics,
]


class VICETrainer(LossFunction):
    def __init__(
        self,
        model,
        positive_buffer,
        negative_buffer,
        batch_size=128,
    ):
        self.model = model
        self.positive_buffer = positive_buffer
        self.negative_buffer = negative_buffer

    def train_epoch(self, epoch, dataset, batches=100):
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
        reconstructions, obs_distribution_params, latent_distribution_params = self.model(obs)
        log_prob = self.model.logprob(obs, obs_distribution_params)
        kle = self.model.kl_divergence(latent_distribution_params)
        loss = -1 * log_prob + beta * kle

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['beta'] = beta
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_statistics[prefix + "log_probs"].append(log_prob.item())
        self.eval_statistics[prefix + "kles"].append(kle.item())

        encoder_mean = self.model.get_encoding_from_latent_distribution_params(latent_distribution_params)
        z_data = ptu.get_numpy(encoder_mean.cpu())
        for i in range(len(z_data)):
            self.eval_data[prefix + "zs"].append(z_data[i, :])
        self.eval_data[prefix + "last_batch"] = (obs, reconstructions)

        return loss

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

    def end_epoch(self, epoch):
        self.eval_statistics = collections.defaultdict(list)
        self.test_last_batch = None

    def get_diagnostics(self):
        stats = OrderedDict()
        for k in sorted(self.eval_statistics.keys()):
            stats[k] = np.mean(self.eval_statistics[k])
        return stats


class VICERewardFn(ContextualRewardFn):
    def __init__(
        self,
        model,
    ):
        self.model = model

    def __call__(self, states, actions, next_states, contexts):
        obs = ptu.from_numpy(next_states["observation"])
        r = self.model(obs)
        return ptu.get_numpy(r)
