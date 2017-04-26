"""
:author: Vitchyr Pong
"""
from collections import OrderedDict

import tensorflow as tf
from typing import Iterable
import numpy as np

from railrl.algos.ddpg import DDPG, TargetUpdateMode
from railrl.core import tf_util
from railrl.core.rnn.rnn import OutputStateRnn
from railrl.data_management.subtraj_replay_buffer import (
    SubtrajReplayBuffer
)
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.policies.memory.rnn_cell_policy import RnnCellPolicy
from railrl.pythonplusplus import map_recursive, filter_recursive
from railrl.qfunctions.nn_qfunction import NNQFunction

TARGET_PREFIX = "target_"


class BpttDDPG(DDPG):
    """
    The ICML idea: this does DDPG updates, but also does BPTT assuming you
    have a recurrent policy.
    """

    def __init__(
            self,
            env,
            exploration_strategy,
            policy: RnnCellPolicy,
            qf: NNQFunction,
            num_bptt_unrolls=1,
            env_obs_dim=None,
            replay_pool_size=10000,
            replay_buffer_class=SubtrajReplayBuffer,
            freeze_hidden=False,
            optimize_simultaneously=False,
            **kwargs
    ):
        """
        :param env: Environment
        :param exploration_strategy: ExplorationStrategy
        :param policy: Policy that is Serializable
        :param qf: QFunctions that is Serializable
        :param qf_learning_rate: Learning rate of the qf
        :param policy_learning_rate: Learning rate of the _policy
        :param qf_weight_decay: How much to decay the weights for Q
        :return:
        """
        self._num_bptt_unrolls = num_bptt_unrolls
        self._env_obs_dim = env_obs_dim
        self._freeze_hidden = freeze_hidden
        self._optimize_simultaneously = optimize_simultaneously

        self._rnn_cell_scope = policy.rnn_cell_scope
        self._rnn_cell = policy.rnn_cell

        self._replay_buffer_class = replay_buffer_class
        self._last_env_scores = []
        super().__init__(
            env,
            exploration_strategy,
            policy,
            qf,
            replay_pool=replay_buffer_class(
                replay_pool_size,
                env,
                num_bptt_unrolls,
            ),
            **kwargs)

    def _sample_minibatch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return self.pool.random_subtrajectories(batch_size)

    def _do_training(
            self,
            epoch=None,
            n_steps_total=None,
            n_steps_current_epoch=None,
    ):
        minibatch = self._sample_minibatch()

        qf_ops = self._get_qf_training_ops()
        qf_feed_dict = self._qf_feed_dict_from_batch(minibatch)
        self.sess.run(qf_ops, feed_dict=qf_feed_dict)

        policy_ops = self._get_policy_training_ops()
        policy_feed_dict = self._policy_feed_dict_from_batch(minibatch)
        self.sess.run(policy_ops, feed_dict=policy_feed_dict)

    def _statistics_from_paths(self, paths) -> OrderedDict:
        eval_pool = self._replay_buffer_class(
            len(paths) * self.max_path_length,
            self.env,
            self._num_bptt_unrolls,
        )
        for path in paths:
            eval_pool.add_trajectory(path)
        batch = eval_pool.get_all_valid_subtrajectories()

        qf_feed_dict = self._qf_feed_dict_from_batch(batch)
        policy_feed_dict = self._policy_feed_dict_from_batch(batch)
        qf_stat_names, qf_ops = zip(*self._qf_statistic_names_and_ops())
        policy_stat_names, policy_ops = zip(
            *self._policy_statistic_names_and_ops())

        statistics = OrderedDict()
        for ops, feed_dict, stat_names in [
            (qf_ops, qf_feed_dict, qf_stat_names),
            (policy_ops, policy_feed_dict, policy_stat_names),
        ]:
            values = self.sess.run(ops, feed_dict=feed_dict)
            for stat_name, value in zip(stat_names, values):
                statistics.update(
                    create_stats_ordered_dict(stat_name, value)
                )

        return statistics

    """
    Q function methods
    """
    def _init_qf_ops(self):
        if not self._optimize_simultaneously:
            return super()._init_qf_ops()

        """
        Backprop the Bellman error through time, i.e. through dQ/dwrite action
        """
        action_input = (self.qf.action_input[0], self._final_rnn_action[1])
        self.qf_with_write_input = self.qf.get_weight_tied_copy(
            action_input=action_input,
        )
        self.ys = (
            self.rewards_placeholder +
            (1. - self.terminals_placeholder)
            * self.discount
            * self.target_qf.output
        )
        self.qf_loss = tf.squeeze(tf_util.mse(self.ys,
                                              self.qf_with_write_input.output))
        self.Q_weights_norm = tf.reduce_sum(
            tf.stack(
                [tf.nn.l2_loss(v)
                 for v in
                 self.qf.get_params_internal(regularizable=True)]
            ),
            name='weights_norm'
        )
        self.qf_total_loss = (
            self.qf_loss + self.qf_weight_decay * self.Q_weights_norm
        )
        self.train_qf_op = tf.train.AdamOptimizer(
            self.qf_learning_rate).minimize(
            self.qf_total_loss,
            var_list=self.qf.get_params() + self.policy.get_params(),
        )

    def _get_qf_training_ops(
            self,
            epoch=None,
            n_steps_total=None,
            n_steps_current_epoch=None,
    ):
        train_ops = [
            self.train_qf_op,
        ]
        if self._batch_norm:
            train_ops += self.qf.batch_norm_update_stats_op

        target_ops = []
        if self._target_update_mode == TargetUpdateMode.SOFT:
            target_ops = [
                self.update_target_qf_op,
            ]
        elif self._target_update_mode == TargetUpdateMode.HARD:
            if n_steps_total % self._hard_update_period == 0:
                target_ops = [
                    self.update_target_qf_op,
                ]
        elif self._target_update_mode == TargetUpdateMode.NONE:
            target_ops = [
                self.update_target_qf_op,
            ]
        else:
            raise RuntimeError(
                "Unknown target update mode: {}".format(
                    self._target_update_mode
                )
            )

        return filter_recursive([
            train_ops,
            target_ops,
        ])

    def _qf_feed_dict_from_batch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        target_numbers = batch['target_numbers']
        times = batch['times']

        flat_actions = actions.reshape(-1, actions.shape[-1])
        flat_obs = obs.reshape(-1, obs.shape[-1])
        flat_next_obs = next_obs.reshape(-1, next_obs.shape[-1])
        flat_target_numbers = target_numbers.flatten()
        flat_times = times.flatten()
        flat_terminals = terminals.flatten()
        flat_rewards = rewards.flatten()

        qf_obs = self._split_flat_obs(flat_obs)
        qf_actions = self._split_flat_actions(flat_actions)
        qf_next_obs = self._split_flat_obs(flat_next_obs)

        feed = self._qf_feed_dict(
            flat_rewards,
            flat_terminals,
            qf_obs,
            qf_actions,
            qf_next_obs,
            target_numbers=flat_target_numbers,
            times=flat_times,
        )
        return feed

    def _qf_statistic_names_and_ops(self):
        return [
            ('QfLoss', self.qf_loss),
            ('QfOutput', self.qf.output),
            # ('OracleQfOutput', self.oracle_qf.output),
        ]

    """
    Policy methods
    """
    def _init_policy_ops(self):
        self._rnn_inputs_ph = tf.placeholder(
            tf.float32,
            [None, self._num_bptt_unrolls, self._env_obs_dim],
            name='rnn_time_inputs',
        )
        self._rnn_inputs_unstacked = tf.unstack(self._rnn_inputs_ph, axis=1)
        self._rnn_init_state_ph = self.policy.get_init_state_placeholder()

        self._rnn_cell_scope.reuse_variables()
        self._save_rnn_cell = OutputStateRnn(
            self._rnn_cell,
        )
        with tf.variable_scope(self._rnn_cell_scope):
            self._rnn_outputs, self._rnn_final_state = tf.contrib.rnn.static_rnn(
                self._save_rnn_cell,
                self._rnn_inputs_unstacked,
                initial_state=self._rnn_init_state_ph,
                dtype=tf.float32,
                scope=self._rnn_cell_scope,
            )
        self._final_rnn_output = self._rnn_outputs[-1][0]
        self._final_rnn_action = (
            self._final_rnn_output,  # pi_a(o, m)
            self._rnn_final_state,  # pi_w(o, m)
        )
        self._final_rnn_action = self._rnn_outputs[-1]
        self._final_rnn_input = (
            self._rnn_inputs_unstacked[-1],  # o
            # self._rnn_outputs[-2][1]       # This is right, but doesn't work
            self.qf.observation_input[1],  # TODO(vitchyr): Why's this better?
        )
        self._final_rnn_input = (
            self._rnn_inputs_unstacked[-1],
            self._rnn_outputs[-2][1]
        )
        self.qf_with_action_input = self.qf.get_weight_tied_copy(
            action_input=self._final_rnn_action,
            observation_input=self._final_rnn_input,
        )
        self.policy_surrogate_loss = - tf.reduce_mean(
            self.qf_with_action_input.output
        )
        if self._freeze_hidden:
            trainable_policy_params = self.policy.get_params(env_only=True)
        else:
            trainable_policy_params = self.policy.get_params_internal()
        self.train_policy_op = tf.train.AdamOptimizer(
            self.policy_learning_rate
        ).minimize(
            self.policy_surrogate_loss,
            var_list=trainable_policy_params,
        )

    def _get_policy_training_ops(
            self,
            epoch=None,
            n_steps_total=None,
            n_steps_current_epoch=None,
    ):
        train_ops = [
            self.train_policy_op,
        ]
        if self._batch_norm:
            train_ops += self.policy.batch_norm_update_stats_op

        target_ops = []
        if self._target_update_mode == TargetUpdateMode.SOFT:
            target_ops = [
                self.update_target_policy_op,
            ]
        elif self._target_update_mode == TargetUpdateMode.HARD:
            if n_steps_total % self._hard_update_period == 0:
                target_ops = [
                    self.update_target_policy_op,
                ]
        elif self._target_update_mode == TargetUpdateMode.NONE:
            target_ops = [
                self.update_target_policy_op,
            ]
        else:
            raise RuntimeError(
                "Unknown target update mode: {}".format(
                    self._target_update_mode
                )
            )

        return filter_recursive([
            train_ops,
            target_ops,
        ])

    def _policy_statistic_names_and_ops(self):
        return [
            ('PolicySurrogateLoss', self.policy_surrogate_loss),
            ('PolicyOutput', self.policy.output),
            # ('OracleQfOutput', self.oracle_qf.output),
        ]

    def _policy_feed_dict_from_batch(self, batch):
        obs = self._split_flat_obs(batch['observations'])
        initial_memory_obs = self._get_time_step(obs, 0)[1]
        env_obs, _ = obs
        last_obs = self._get_time_step(obs, -1)
        return {
            # self.qf_with_action_input.observation_input: last_obs,
            self._rnn_inputs_ph: env_obs,
            self._rnn_init_state_ph: initial_memory_obs,
            # self.policy.observation_input: last_obs,  # this is for eval to work
        }


    """
    Miscellaneous functions
    """
    @staticmethod
    def _get_time_step(action_or_obs, t):
        """
        Squeeze time out by only taking the one time step.

        :param action_or_obs: tuple of Tensors or Tensor of shape [batch size x
        traj length x dim]
        :param t: The time index to slice out.
        :return: return tuple of Tensors or Tensor of shape [batch size x dim]
        """
        return map_recursive(lambda x: x[:, t, :], action_or_obs)

    def log_diagnostics(self, paths):
        self._last_env_scores.append(np.mean(self.env.log_diagnostics(paths)))
        self.policy.log_diagnostics(paths)

    @property
    def epoch_scores(self) -> Iterable[float]:
        """
        :return: The scores after each epoch training. The objective is to
        MAXIMIZE these value.
        """
        return self._last_env_scores
