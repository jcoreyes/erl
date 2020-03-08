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
from railrl.misc.ml_util import PiecewiseLinearSchedule, ConstantSchedule
import torch.nn.functional as F
from railrl.torch.networks import LinearTransform

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
            beta_schedule_kwargs=None,

            policy_lr=1e-3,
            qf_lr=1e-3,
            policy_weight_decay=0,
            q_weight_decay=0,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,

            bc_num_pretrain_steps=0,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=0,
            bc_batch_size=128,
            bc_loss_type="mle",
            awr_loss_type="mle",
            save_bc_policies=0,
            alpha=1.0,

            policy_update_period=1,
            q_update_period=1,

            weight_loss=True,
            compute_bc=True,

            bc_weight=0.0,
            rl_weight=1.0,
            use_awr_update=True,
            reparam_weight=1.0,
            awr_weight=1.0,
            post_pretrain_hyperparams=None,
            post_bc_pretrain_hyperparams=None,

            awr_use_mle_for_vf=False,
            awr_sample_actions=False,
            awr_min_q=False,

            reward_transform_class=None,
            reward_transform_kwargs=None,
            terminal_transform_class=None,
            terminal_transform_kwargs=None,

            pretraining_env_logging_period=100000,
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

        self.awr_use_mle_for_vf = awr_use_mle_for_vf
        self.awr_sample_actions = awr_sample_actions
        self.awr_min_q = awr_min_q

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
            weight_decay=q_weight_decay,
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            weight_decay=q_weight_decay,
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.beta = beta
        self.beta_schedule_kwargs = beta_schedule_kwargs
        if beta_schedule_kwargs is None:
            self.beta_schedule = ConstantSchedule(beta)
        else:
            schedule_class = beta_schedule_kwargs.pop("schedule_class", PiecewiseLinearSchedule)
            self.beta_schedule = schedule_class(**beta_schedule_kwargs)

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.bc_num_pretrain_steps = bc_num_pretrain_steps
        self.q_num_pretrain1_steps = q_num_pretrain1_steps
        self.q_num_pretrain2_steps = q_num_pretrain2_steps
        self.bc_batch_size = bc_batch_size
        self.bc_loss_type = bc_loss_type
        self.awr_loss_type = awr_loss_type
        self.rl_weight = rl_weight
        self.bc_weight = bc_weight
        self.save_bc_policies = save_bc_policies
        self.eval_policy = MakeDeterministic(self.policy)
        self.compute_bc = compute_bc
        self.alpha = alpha
        self.q_update_period = q_update_period
        self.policy_update_period = policy_update_period
        self.weight_loss = weight_loss

        self.reparam_weight = reparam_weight
        self.awr_weight = awr_weight
        self.post_pretrain_hyperparams = post_pretrain_hyperparams
        self.post_bc_pretrain_hyperparams = post_bc_pretrain_hyperparams
        self.update_policy = True
        self.pretraining_env_logging_period = pretraining_env_logging_period

        self.reward_transform_class = reward_transform_class or LinearTransform
        self.reward_transform_kwargs = reward_transform_kwargs or dict(m=1, b=0)
        self.terminal_transform_class = terminal_transform_class or LinearTransform
        self.terminal_transform_kwargs = terminal_transform_kwargs or dict(m=1, b=0)
        self.reward_transform = self.reward_transform_class(**self.reward_transform_kwargs)
        self.terminal_transform = self.terminal_transform_class(**self.terminal_transform_kwargs)


    def get_batch_from_buffer(self, replay_buffer):
        batch = replay_buffer.random_batch(self.bc_batch_size)
        batch = np_to_pytorch_batch(batch)
        return batch

    def run_bc_batch(self, replay_buffer):
        batch = self.get_batch_from_buffer(replay_buffer)
        o = batch["observations"]
        u = batch["actions"]
        # g = batch["resampled_goals"]
        # og = torch.cat((o, g), dim=1)
        og = o
        # pred_u, *_ = self.policy(og)
        pred_u, policy_mean, policy_log_std, log_pi, entropy, policy_std, mean_action_log_prob, pretanh_value, dist = self.policy(
            og, deterministic=False, reparameterize=True, return_log_prob=True,
        )

        mse = (pred_u - u) ** 2
        mse_loss = mse.mean()

        policy_logpp = dist.log_prob(u, )
        logp_loss = -policy_logpp.mean()

        # T = 0
        if self.bc_loss_type == "mle":
            policy_loss = logp_loss
        elif self.bc_loss_type == "mse":
            policy_loss = mse_loss
        else:
            error

        return policy_loss, logp_loss, mse_loss, policy_log_std

    def pretrain_policy_with_bc(self):
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'pretrain_policy.csv', relative_to_snapshot_dir=True
        )
        total_ret = 0
        for _ in range(20):
            o = self.env.reset()
            ret = 0
            for _ in range(1000):
                a, _ = self.policy.get_action(o)
                o, r, done, info = self.env.step(a)
                ret += r
                if done:
                    break
            total_ret += ret
        print("INITIAL RETURN", total_ret/20)

        for i in range(self.bc_num_pretrain_steps):
            train_policy_loss, train_logp_loss, train_mse_loss, train_log_std = self.run_bc_batch(self.demo_train_buffer)
            train_policy_loss = train_policy_loss * self.bc_weight

            self.policy_optimizer.zero_grad()
            train_policy_loss.backward()
            self.policy_optimizer.step()

            test_policy_loss, test_logp_loss, test_mse_loss, test_log_std = self.run_bc_batch(self.demo_test_buffer)
            test_policy_loss = test_policy_loss * self.bc_weight
            
            if i % self.pretraining_env_logging_period == 0:
                total_ret = 0
                for _ in range(20):
                    o = self.env.reset()
                    ret = 0
                    for _ in range(1000):
                        a, _ = self.policy.get_action(o)
                        o, r, done, info = self.env.step(a)
                        ret += r
                        if done:
                            break
                    total_ret += ret
                print("Return at step {} : {}".format(i, total_ret/20))
            if i % 1000 == 0:
                stats = {
                "pretrain_bc/batch": i,
                "pretrain_bc/avg_return": total_ret / 20,
                "pretrain_bc/Train Logprob Loss": ptu.get_numpy(train_logp_loss),
                "pretrain_bc/Test Logprob Loss": ptu.get_numpy(test_logp_loss),
                "pretrain_bc/Train MSE": ptu.get_numpy(train_mse_loss),
                "pretrain_bc/Test MSE": ptu.get_numpy(test_mse_loss),
                "pretrain_bc/train_policy_loss": ptu.get_numpy(train_policy_loss),
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

        if self.post_bc_pretrain_hyperparams:
            self.set_algorithm_weights(**self.post_bc_pretrain_hyperparams)

    def pretrain_q_with_bc_data(self):
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'pretrain_q.csv', relative_to_snapshot_dir=True
        )

        self.update_policy = False
        # first train only the Q function
        for i in range(self.q_num_pretrain1_steps):
            self.eval_statistics = dict()
            if i % self.pretraining_env_logging_period == 0:
                self._need_to_update_eval_statistics = True

            train_data = self.replay_buffer.random_batch(self.bc_batch_size)
            train_data = np_to_pytorch_batch(train_data)
            obs = train_data['observations']
            next_obs = train_data['next_observations']
            # goals = train_data['resampled_goals']
            train_data['observations'] = obs # torch.cat((obs, goals), dim=1)
            train_data['next_observations'] = next_obs # torch.cat((next_obs, goals), dim=1)
            self.train_from_torch(train_data)
            if i % 1000 == 0:
                logger.record_dict(self.eval_statistics)
                logger.dump_tabular(with_prefix=True, with_timestamp=False)

        self.update_policy = True
        # then train policy and Q function together
        for i in range(self.q_num_pretrain2_steps):
            self.eval_statistics = dict()
            train_data = self.replay_buffer.random_batch(self.bc_batch_size)
            train_data = np_to_pytorch_batch(train_data)
            obs = train_data['observations']
            next_obs = train_data['next_observations']
            # goals = train_data['resampled_goals']
            train_data['observations'] = obs # torch.cat((obs, goals), dim=1)
            train_data['next_observations'] = next_obs # torch.cat((next_obs, goals), dim=1)
            self.train_from_torch(train_data)

            if i % self.pretraining_env_logging_period == 0:
                total_ret = 0
                for _ in range(20):
                    o = self.env.reset()
                    ret = 0
                    for _ in range(1000):
                        a, _ = self.policy.get_action(o)
                        o, r, done, info = self.env.step(a)
                        ret += r
                        if done:
                            break
                    total_ret += ret
                print("Return at step {} : {}".format(i, total_ret/20))
            
            if self._need_to_update_eval_statistics:
                self.eval_statistics["avg_return"] =  total_ret / 20
                self.eval_statistics["batch"] = i
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

        self._need_to_update_eval_statistics = True
        self.eval_statistics = dict()

        if self.post_pretrain_hyperparams:
            self.set_algorithm_weights(**self.post_pretrain_hyperparams)

    def set_algorithm_weights(
        self,
        # bc_weight,
        # rl_weight,
        # use_awr_update,
        # use_reparam_update,
        # reparam_weight,
        # awr_weight,
        **kwargs
    ):
        for key in kwargs:
            self.__dict__[key] = kwargs[key]
        # self.bc_weight = bc_weight
        # self.rl_weight = rl_weight
        # self.use_awr_update = use_awr_update
        # self.use_reparam_update = use_reparam_update
        # self.awr_weight = awr_weight

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        weights = batch.get('weights', None)

        if self.reward_transform:
            rewards = self.reward_transform(rewards)

        if self.terminal_transform:
            terminals = self.terminal_transform(terminals)

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, entropy, policy_std, mean_action_log_prob, pretanh_value, dist = self.policy(
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
            alpha = self.alpha

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

        """
        Policy Loss
        """
        qf1_new_actions = self.qf1(obs, new_obs_actions)
        qf2_new_actions = self.qf2(obs, new_obs_actions)
        q_new_actions = torch.min(
            qf1_new_actions,
            qf2_new_actions,
        )

        # Advantage-weighted regression
        if self.awr_use_mle_for_vf:
            v_pi = self.qf1(obs, policy_mean)
        else:
            v_pi = self.qf1(obs, new_obs_actions)

        if self.awr_sample_actions:
            u = new_obs_actions
            if self.awr_min_q:
                q_adv = q_new_actions
            else:
                q_adv = qf1_new_actions
        else:
            u = actions
            if self.awr_min_q:
                q_adv = torch.min(q1_pred, q2_pred)
            else:
                q_adv = q1_pred

        if self.awr_loss_type == "mse":
            policy_logpp = -(policy_mean - u) ** 2
        else:
            policy_logpp = dist.log_prob(u)
            policy_logpp = policy_logpp.sum(dim=1, keepdim=True)

        advantage = q_adv - v_pi

        if self.weight_loss and weights is None:
            beta = self.beta_schedule.get_value(self._n_train_steps_total)
            weights = F.softmax(advantage / beta, dim=0)

        policy_loss = alpha * log_pi.mean()

        if self.use_awr_update and self.weight_loss:
            policy_loss = policy_loss + self.awr_weight * (-policy_logpp * len(weights)*weights.detach()).mean()
        elif self.use_awr_update:
            policy_loss = policy_loss + self.awr_weight * (-policy_logpp).mean()

        if self.reparam_weight:
            policy_loss = policy_loss + self.reparam_weight * (-q_new_actions).mean()
        else:
            policy_loss = policy_loss - q_new_actions.mean()

        policy_loss = self.rl_weight * policy_loss
        if self.compute_bc:
            train_policy_loss, train_logp_loss, train_mse_loss, _ = self.run_bc_batch(self.demo_train_buffer)
            policy_loss = policy_loss + self.bc_weight * train_policy_loss

        """
        Update networks
        """
        if self._n_train_steps_total % self.q_update_period == 0:
            self.qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer.step()

        if self._n_train_steps_total % self.policy_update_period == 0 and self.update_policy:
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

            if self.compute_bc:
                test_policy_loss, test_logp_loss, test_mse_loss, _ = self.run_bc_batch(self.demo_test_buffer)
                self.eval_statistics.update({
                    "bc/Train Logprob Loss": ptu.get_numpy(train_logp_loss),
                    "bc/Test Logprob Loss": ptu.get_numpy(test_logp_loss),
                    "bc/Train MSE": ptu.get_numpy(train_mse_loss),
                    "bc/Test MSE": ptu.get_numpy(test_mse_loss),
                    "bc/train_policy_loss": ptu.get_numpy(train_policy_loss),
                    "bc/test_policy_loss": ptu.get_numpy(test_policy_loss),
                })
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
