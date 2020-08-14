import warnings
from typing import Any, Callable, Dict, List

import numpy as np
from gym.spaces import Box, Dict
from multiworld.core.multitask_env import MultitaskEnv

from rlkit import pythonplusplus as ppp
from rlkit.core.distribution import DictDistribution
from rlkit.envs.contextual import ContextualRewardFn
from rlkit.envs.contextual.contextual_env import (
    ContextualDiagnosticsFn,
    Path,
    Context,
    Diagnostics,
)
from rlkit.envs.images import Renderer
from rlkit.core.loss import LossFunction
from rlkit.torch import pytorch_util as ptu

import torch
from torch import optim
import collections
from collections import OrderedDict

Observation = Dict
Goal = Any
GoalConditionedDiagnosticsFn = Callable[
    [List[Path], List[Goal]],
    Diagnostics,
]


class AIRLTrainer(LossFunction):
    def __init__(
        self,
        model,
        positives,
        policy,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=128,
        data_split=1,
        train_split=0.9,
    ):
        """positives are a 2D numpy array"""

        self.model = model
        self.positives = positives
        self.policy = policy
        self.batch_size = batch_size
        self.epoch = 0

        # self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.softmax = torch.nn.Softmax()

        self.lr = lr
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params,
            lr=self.lr,
            weight_decay=weight_decay,
        )

        # stateful tracking variables, reset every epoch
        self.eval_statistics = collections.defaultdict(list)
        self.eval_data = collections.defaultdict(list)

    def compute_loss(self, batch, epoch=-1, test=False):
        prefix = "test/" if test else "train/"

        positives = self.positives.random_batch(self.batch_size)["observations"]
        P, feature_size = positives.shape
        negatives = batch['observations']
        N, feature_size = negatives.shape

        X = np.concatenate((positives, negatives))
        Y = np.zeros((P + N, 2))
        Y[:P, 0] = 1
        Y[P:, 1] = 1

        X = ptu.from_numpy(X)
        Y = ptu.from_numpy(Y)
        y_pred = self.softmax(self.airl_discriminator_logits(X)) # self.model(X) # todo: logsumexp

        # import ipdb; ipdb.set_trace()
        loss = self.loss_fn(y_pred, Y)

        y_pred_class = (y_pred > 0).float()

        self.eval_statistics[prefix + "tp"].append((y_pred_class * Y).mean().item())
        self.eval_statistics[prefix + "tn"].append(((1 - y_pred_class) * (1 - Y)).mean().item())
        self.eval_statistics[prefix + "fn"].append(((1 - y_pred_class) * Y).mean().item())
        self.eval_statistics[prefix + "fp"].append((y_pred_class * (1 - Y)).mean().item())

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics[prefix + "losses"].append(loss.item())

        return loss

    def airl_discriminator_logits(self, observations):
        log_p = self.model(observations)
        dist = self.policy(observations)
        new_obs_actions, log_pi = dist.sample_and_logprob()
        logits = torch.cat((log_p, log_pi[:, None]), dim=1)
        return logits

    def train(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.compute_loss(batch, self.epoch, False)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.model.eval()
        self.compute_loss(batch, self.epoch, True)

        self.epoch += 1

    def get_batch(self, test=False):
        if test:
            indices = np.random.randint(self.train_N, self.data_N, self.batch_size)
        else:
            indices = np.random.randint(0, self.train_N, self.batch_size)
        return self.positives[indices, :]

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


class AIRLRewardFn(ContextualRewardFn):
    def __init__(
        self,
        model,
        context_keys=None
    ):
        self.model = model
        self.context_keys = context_keys or []

    def __call__(self, states, actions, next_states, contexts):
        contexts = [contexts[k] for k in self.context_keys]
        full_obs = [next_states["observation"], ] + contexts
        np_obs = np.concatenate(full_obs, axis=1)
        obs = ptu.from_numpy(np_obs)
        r = self.model(obs)
        return ptu.get_numpy(r)
