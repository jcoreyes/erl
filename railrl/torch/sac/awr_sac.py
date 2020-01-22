from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from railrl.torch.sac.policies import MakeDeterministic
from torch import nn as nn

import railrl.torch.pytorch_util as ptu
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.torch.core import np_to_pytorch_batch
from railrl.torch.torch_rl_algorithm import TorchTrainer
from railrl.core import logger

import torch.nn.functional as F

class AWRSACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,
            beta=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            policy_weight_decay=0,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            path_loader=None,

            use_awr_update=True,
            bc_num_pretrain_steps=0,
            q_num_pretrain_steps=0,
            bc_batch_size=128,
            bc_loss_type="mle",
            save_bc_policies=0,
            rl_weight=1.0,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_awr_update = use_awr_update
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            weight_decay=policy_weight_decay,
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.beta = beta
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.bc_num_pretrain_steps = bc_num_pretrain_steps
        self.q_num_pretrain_steps = q_num_pretrain_steps
        self.bc_batch_size = bc_batch_size
        self.bc_loss_type = bc_loss_type
        self.rl_weight = rl_weight
        self.save_bc_policies = save_bc_policies
        self.eval_policy = MakeDeterministic(self.policy)

    def get_batch_from_buffer(self, replay_buffer):
        batch = replay_buffer.random_batch(self.bc_batch_size)
        batch = np_to_pytorch_batch(batch)
        return batch

    def pretrain_policy_with_bc(self):
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'pretrain_policy.csv', relative_to_snapshot_dir=True
        )
        for i in range(self.bc_num_pretrain_steps):
            train_batch = self.get_batch_from_buffer(self.demo_train_buffer)
            train_o = train_batch["observations"]
            train_u = train_batch["actions"]
            # train_g = train_batch["resampled_goals"]
            # train_og = torch.cat((train_o, train_g), dim=1)
            train_og = train_o
            # train_pred_u, *_ = self.policy(train_og)
            train_pred_u, policy_mean, policy_log_std, log_pi, entropy, policy_std, *_ = self.policy(
                train_og, deterministic=True, reparameterize=True, return_log_prob=True,
            )
            train_mse = (train_pred_u - train_u) ** 2
            train_mse_loss = train_mse.mean()

            train_policy_logpp = self.policy.logprob(train_u, policy_mean, policy_std)[:, 0]

            # T = 0
            if self.bc_loss_type == "mle":
                policy_loss = -train_policy_logpp.mean()
            elif self.bc_loss_type == "mse":
                policy_loss = train_mse_loss.mean()
            else:
                error
            # if i < T:
            #     policy_loss = train_mse_loss.mean()
            # else:
            #     policy_loss = -train_policy_logpp.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            test_batch = self.get_batch_from_buffer(self.demo_test_buffer)
            test_o = test_batch["observations"]
            test_u = test_batch["actions"]
            # test_g = test_batch["resampled_goals"]
            # test_og = torch.cat((test_o, test_g), dim=1)
            test_og = test_o
            # test_pred_u, *_ = self.policy(test_og)
            test_pred_u, policy_mean, policy_log_std, log_pi, entropy, policy_std, *_ = self.policy(
                test_og, deterministic=True, reparameterize=True, return_log_prob=True,
            )
            test_mse = (test_pred_u - test_u) ** 2
            test_mse_loss = test_mse.mean()

            test_policy_logpp = self.policy.logprob(test_u, policy_mean, policy_std)[:, 0]

            # if i < T:
            #     test_policy_loss = test_mse_loss.mean()
            # else:
            # test_policy_loss = -test_policy_logpp.mean()
            if self.bc_loss_type == "mle":
                test_policy_loss = -test_policy_logpp.mean()
            elif self.bc_loss_type == "mse":
                test_policy_loss = test_mse_loss.mean()
            else:
                error

            train_mse_mean = np.mean(ptu.get_numpy(train_mse_loss))
            test_mse_mean = np.mean(ptu.get_numpy(test_mse_loss))

            train_logp = np.mean(ptu.get_numpy(train_policy_logpp))
            test_logp = np.mean(ptu.get_numpy(test_policy_logpp))

            #
            if i % 100 == 0:
                total_ret = 0
                for _ in range(5):
                    o = self.env.reset()
                    ret = 0
                    for i in range(1000):
                        a, _ = self.eval_policy.get_action(o)
                        o, r, done, info = self.env.step(a)
                        ret += r
                        if done:
                            break
                total_ret += ret
                stats = {
                    "pretrain_bc/avg_return":total_ret/5,
                    "pretrain_bc/batch": i,
                    "pretrain_bc/Train Logprob": train_logp,
                    "pretrain_bc/Test Logprob": test_logp,
                    "pretrain_bc/Train MSE": train_mse_mean,
                    "pretrain_bc/Test MSE": test_mse_mean,
                    "pretrain_bc/train_policy_loss": ptu.get_numpy(policy_loss),
                    "pretrain_bc/test_policy_loss": ptu.get_numpy(test_policy_loss),
                }
                logger.record_dict(stats)
                logger.dump_tabular(with_prefix=True, with_timestamp=False)

            if self.save_bc_policies and i % self.save_bc_policies == 0:
                logger.save_itr_params(i, {
                    "evaluation/policy": self.policy,
                    "evaluation/env": self.env,
                })
        logger.remove_tabular_output(
            'pretrain_policy.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )

    def pretrain_q_with_bc_data(self):
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'pretrain_q.csv', relative_to_snapshot_dir=True
        )

        self.update_policy = False
        # first train only the Q function
        for i in range(self.q_num_pretrain_steps):
            self.eval_statistics = dict()
            self._need_to_update_eval_statistics = True

            train_data = self.replay_buffer.random_batch(self.bc_batch_size)
            train_data = np_to_pytorch_batch(train_data)
            obs = train_data['observations']
            next_obs = train_data['next_observations']
            # goals = train_data['resampled_goals']
            train_data['observations'] = obs # torch.cat((obs, goals), dim=1)
            train_data['next_observations'] = next_obs # torch.cat((next_obs, goals), dim=1)
            self.train_from_torch(train_data)

            logger.record_dict(self.eval_statistics)
            logger.dump_tabular(with_prefix=True, with_timestamp=False)

        self.update_policy = True
        # then train policy and Q function together
        for i in range(self.q_num_pretrain_steps):
            self.eval_statistics = dict()
            self._need_to_update_eval_statistics = True

            train_data = self.replay_buffer.random_batch(self.bc_batch_size)
            train_data = np_to_pytorch_batch(train_data)
            obs = train_data['observations']
            next_obs = train_data['next_observations']
            # goals = train_data['resampled_goals']
            train_data['observations'] = obs # torch.cat((obs, goals), dim=1)
            train_data['next_observations'] = next_obs # torch.cat((next_obs, goals), dim=1)
            self.train_from_torch(train_data)

            logger.record_dict(self.eval_statistics)
            logger.dump_tabular(with_prefix=True, with_timestamp=False)

        logger.remove_tabular_output(
            'pretrain_q.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, entropy, policy_std, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        # Advantage-weighted regression
        v_pi = self.qf1(obs, new_obs_actions)
        # policy_logpp = -(new_obs_actions - actions) ** 2
        # policy_logpp = policy_logpp.mean(dim=1)
        policy_logpp = self.policy.logprob(actions, policy_mean, policy_std)

        advantage = q1_pred - v_pi
        weights = F.softmax(advantage / self.beta, dim=0)

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        if self.use_awr_update:
            policy_loss = self.rl_weight * (alpha*log_pi - policy_logpp * weights.detach()).mean()
        else:
            policy_loss = self.rl_weight * (alpha*log_pi - q_new_actions).mean()

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Advantage Weights',
                ptu.get_numpy(weights),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )
