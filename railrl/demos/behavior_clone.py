from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import railrl.torch.pytorch_util as ptu
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm
from railrl.misc.asset_loader import load_local_or_remote_file

import abc
import copy
import pickle
import time
from collections import OrderedDict

import gtimer as gt
import numpy as np

from railrl.core import logger
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.path_builder import PathBuilder
from railrl.envs.remote import RemoteRolloutEnv
from railrl.misc import eval_util
from railrl.policies.base import ExplorationPolicy
from railrl.samplers.in_place import InPlacePathSampler
from railrl.samplers.rollout_functions import rollout
import railrl.envs.env_utils as env_utils

class BehaviorClone(TorchRLAlgorithm):
    """
    Behavior cloning implementation
    """

    def __init__(
            self,
            env,
            exploration_policy,
            policy,
            demo_path,
            policy_learning_rate=1e-3,
            weight_decay=1e-4,
            optimizer_class=optim.Adam,

            **kwargs
    ):
        super().__init__(
            env,
            exploration_policy=exploration_policy,
            eval_policy=policy,
            **kwargs
        )
        self.policy = policy
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_learning_rate,
            weight_decay=weight_decay,
        )

        self.demo_path = demo_path
        self.load_demos(self.demo_path)

    def load_demos(self, path):
        data = load_local_or_remote_file(self.demo_path)

        for path in data:
            for (
                ob,
                action,
                reward,
                next_ob,
                terminal,
                agent_info,
                env_info,
            ) in zip(
                path["observations"],
                path["actions"],
                path["rewards"],
                path["next_observations"],
                path["terminals"],
                path["agent_infos"],
                path["env_infos"],
            ):
                # goal = path["goal"]["state_desired_goal"][0, :]
                # import pdb; pdb.set_trace()
                # print(goal.shape, ob["state_observation"])
                # state_observation = np.concatenate((ob["state_observation"], goal))
                action = action[:2]
                reward = np.array([reward])
                terminal = np.array([terminal])
                self._handle_step(
                    ob,
                    action,
                    reward,
                    next_ob,
                    terminal,
                    agent_info=agent_info,
                    env_info=env_info,
                )
            self._handle_rollout_ending()

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        self._train_given_data(
            rewards,
            terminals,
            obs,
            actions,
            next_obs,
        )

    def _train_given_data(
        self,
        rewards,
        terminals,
        obs,
        actions,
        next_obs,
        logger_prefix="",
    ):
        """
        Critic operations.
        """

        predicted_actions = self.policy(next_obs)
        error = (predicted_actions - actions) ** 2
        bc_loss = error.mean()

        """
        Update Networks
        """
        self.policy_optimizer.zero_grad()
        bc_loss.backward()
        self.policy_optimizer.step()

        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False

            self.eval_statistics[logger_prefix + 'Policy Loss'] = np.mean(ptu.get_numpy(
                bc_loss
            ))

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        Modified here to always evaluate
        """
        return True

    def evaluate(self, epoch, eval_paths=None):
        statistics = OrderedDict()
        statistics.update(self.eval_statistics)

        logger.log("Collecting samples for evaluation")
        if eval_paths:
            test_paths = eval_paths
        else:
            test_paths = self.get_eval_paths()
        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="Test",
        ))
        # if len(self._exploration_paths) > 0:
        #     statistics.update(eval_util.get_generic_path_information(
        #         self._exploration_paths, stat_prefix="Exploration",
        #     ))
        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths, logger=logger)
        if hasattr(self.env, "get_diagnostics"):
            statistics.update(self.env.get_diagnostics(test_paths))

        average_returns = eval_util.get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns
        for key, value in statistics.items():
            logger.record_tabular(key, value)
        self.need_to_update_eval_statistics = True

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        self.update_epoch_snapshot(snapshot)
        return snapshot

    def update_epoch_snapshot(self, snapshot):
        snapshot.update(
            policy=self.eval_policy,
            trained_policy=self.policy,
            exploration_policy=self.exploration_policy,
        )

    @property
    def networks(self):
        return [
            self.policy,
        ]
