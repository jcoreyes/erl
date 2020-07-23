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

import torch
from torch import optim
import collections

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
        positives,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=30,
    ):
        """positives are a 2D numpy array"""

        self.model = model
        self.positives = positives
        self.batch_size = batch_size
        self.feature_size = positives.shape[1]
        self.epoch = 0

        self.loss_fn = torch.nn.BCEWithLogitsLoss()

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

        X = np.zeros((2 * self.batch_size, self.feature_size))
        Y = np.zeros((2 * self.batch_size, 1))
        X[:self.batch_size] = batch['observations'][:self.batch_size, :self.feature_size]
        Y[:self.batch_size] = 0
        X[self.batch_size:] = self.positives
        Y[self.batch_size:] = 1

        X = ptu.from_numpy(X)
        Y = ptu.from_numpy(Y)
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, Y)

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics[prefix + "losses"].append(loss.item())

        return loss

    def train(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.compute_loss(batch, self.epoch, False)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epoch += 1

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
