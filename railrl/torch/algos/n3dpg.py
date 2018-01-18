from collections import OrderedDict

import numpy as np
import torch.optim as optim

import railrl.torch.pytorch_util as ptu
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.ml_util import ConstantSchedule
from railrl.torch.algos.torch_rl_algorithm import TorchRLAlgorithm
from railrl.torch.modules import HuberLoss


class N3DPG(TorchRLAlgorithm):
    """
    Like DDPG but have 3 networks:
    1. Q
    2. V
    3. Policy
    """

    def __init__(
            self,
            env,
            qf,
            vf,
            policy,
            exploration_policy,

            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
            qf_weight_decay=0,
            qf_criterion=None,
            vf_learning_rate=1e-3,
            vf_criterion=None,
            epoch_discount_schedule=None,

            target_hard_update_period=1000,
            tau=1e-2,
            use_soft_update=False,

            plotter=None,
            render_eval_paths=False,

            **kwargs
    ):
        super().__init__(
            env,
            exploration_policy,
            eval_policy=policy,
            **kwargs
        )
        if qf_criterion is None:
            qf_criterion = HuberLoss()
        if vf_criterion is None:
            vf_criterion = HuberLoss()
        self.qf = qf
        self.vf = vf
        self.policy = policy
        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.qf_weight_decay = qf_weight_decay
        self.qf_criterion = qf_criterion
        self.vf_learning_rate = vf_learning_rate
        self.vf_criterion = vf_criterion
        if epoch_discount_schedule is None:
            epoch_discount_schedule = ConstantSchedule(self.discount)

        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update

        self.epoch_discount_schedule = epoch_discount_schedule
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.target_vf = self.vf.copy()
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.qf_learning_rate,
        )
        self.vf_optimizer = optim.Adam(
            self.vf.parameters(),
            lr=self.vf_learning_rate,
        )
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.policy_learning_rate,
        )
        self.eval_statistics = None

    def _start_epoch(self, epoch):
        super()._start_epoch(epoch)
        self.discount = self.epoch_discount_schedule.get_value(epoch)

    def _do_training(self):
        batch = self.get_batch(training=True)
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy operations.
        """
        policy_actions = self.policy(obs)
        q_output = self.qf(obs, policy_actions)
        policy_loss = - q_output.mean()

        """
        Qf operations.
        """
        target_q_values = self.target_vf(next_obs)
        q_target = rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()
        q_pred = self.qf(obs, actions)
        bellman_errors = (q_pred - q_target) ** 2
        qf_loss = self.qf_criterion(q_pred, q_target)

        """
        Vf operations.
        """
        v_target = self.qf(next_obs, self.policy(next_obs)).detach()
        v_pred = self.vf(next_obs)
        vf_loss = self.vf_criterion(v_pred, v_target)

        """
        Update Networks
        """

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self._update_target_networks()

        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Targets',
                ptu.get_numpy(v_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors',
                ptu.get_numpy(bellman_errors),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))

    def _update_target_networks(self):
        if self.use_soft_update:
            ptu.soft_update_from_to(self.vf, self.target_vf, self.tau)
        else:
            if self._n_env_steps_total % self.target_hard_update_period == 0:
                ptu.copy_model_params_from_to(self.vf, self.target_vf)

    def offline_evaluate(self, epoch):
        raise NotImplementedError()

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            policy=self.policy,
            env=self.training_env,
            exploration_policy=self.exploration_policy,
            qf=self.qf,
            vf=self.vf,
            batch_size=self.batch_size,
        )

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.vf,
        ]
