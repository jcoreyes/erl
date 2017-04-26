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
from railrl.pythonplusplus import map_recursive, filter_recursive, line_logger
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
            bpt_bellman_error_weight=0.,
            train_policy=True,
            extra_train_period=100,
            num_extra_qf_updates=0,
            extra_qf_training_mode='none',
            qf_total_loss_tolerance=None,
            max_num_q_updates=100,
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
        :param extra_qf_training_mode: String:
         - 'none' : Don't do any extra QF training
         - 'fixed': Always do `num_extra_qf_updates` extra updates
         - 'validation': Do up to `max_num_q_updates` extra updates so long
         as validation qf loss goes down.
        :return:
        """
        assert extra_qf_training_mode in [
            'none',
            'fixed',
            'validation',
        ]
        self._num_bptt_unrolls = num_bptt_unrolls
        self._env_obs_dim = env_obs_dim
        self._freeze_hidden = freeze_hidden
        self._bpt_bellman_error_weight = bpt_bellman_error_weight
        self.train_policy = train_policy
        self.extra_qf_training_mode = extra_qf_training_mode
        self.extra_train_period = extra_train_period
        self._num_extra_qf_updates = num_extra_qf_updates
        self.qf_total_loss_tolerance = qf_total_loss_tolerance
        self.max_num_q_updates = max_num_q_updates

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
        self._do_extra_qf_training(n_steps_total=n_steps_total)

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

        """
        The batch is a bunch of subsequences. Flatten the subsequences so
        that they just look like normal (s, a, s') tuples.
        """
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
    Extra QF Training Functions
    """
    def _do_extra_qf_training(self, n_steps_total=None, **kwargs):
        if self.extra_qf_training_mode == 'none':
            return
        elif self.extra_qf_training_mode == 'fixed':
            for _ in range(self._num_extra_qf_updates):
                minibatch = self._sample_minibatch()
                feed_dict = self._qf_feed_dict_from_batch(minibatch)
                ops = self._get_qf_training_ops(n_steps_total=0)
                if len(ops) > 0:
                    self.sess.run(ops, feed_dict=feed_dict)
        elif self.extra_qf_training_mode == 'validation':
            if self.max_num_q_updates <= 0:
                return

            last_validation_loss = self._validation_qf_loss()
            if self._should_train_qf_extra(n_steps_total=n_steps_total):
                line_logger.print_over(
                    "{0} T:{1:3.4f} V:{2:3.4f}".format(0, 0, 0)
                )
                for i in range(self.max_num_q_updates):
                    minibatch = self._sample_minibatch()
                    feed_dict = self._qf_feed_dict_from_batch(minibatch)
                    ops = [self.qf_loss, self.train_qf_op]
                    ops += self.update_target_qf_op
                    train_loss = float(
                        self.sess.run(ops, feed_dict=feed_dict)[0]
                    )
                    validation_loss = self._validation_qf_loss()
                    line_logger.print_over(
                        "{0} T:{1:3.4f} V:{2:3.4f}".format(
                            i, train_loss, validation_loss,
                        )
                    )
                    if validation_loss > last_validation_loss:
                        break
                    if validation_loss <= self.qf_total_loss_tolerance:
                        break
                    last_validation_loss = validation_loss
                line_logger.newline()

    def _validation_qf_loss(self):
        batch = self.pool.get_valid_subtrajectories(validation=True)
        feed_dict = self._qf_feed_dict_from_batch(batch)
        return self.sess.run(self.qf_loss, feed_dict=feed_dict)

    def _should_train_qf_extra(self, n_steps_total):
        return (
            True
            and n_steps_total % self.extra_train_period == 0
            and self.train_qf_op is not None
            and self.qf_total_loss_tolerance is not None
            and self.max_num_q_updates > 0
        )


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
        self._rnn_outputs, self._rnn_final_state = tf.contrib.rnn.static_rnn(
            self._save_rnn_cell,
            self._rnn_inputs_unstacked,
            initial_state=self._rnn_init_state_ph,
            dtype=tf.float32,
            scope=self._rnn_cell_scope,
        )
        self._final_rnn_augmented_action = self._rnn_outputs[-1]
        self._final_rnn_memory_input = self._rnn_outputs[-2][1]
        self._final_rnn_augmented_input = (
            self._rnn_inputs_unstacked[-1],
            self._final_rnn_memory_input,
        )
        self.qf_with_action_input = self.qf.get_weight_tied_copy(
            action_input=self._final_rnn_augmented_action,
            observation_input=self._final_rnn_augmented_input,
        )

        """
        Backprop the Bellman error through time, i.e. through dQ/dwrite action
        """
        if self._bpt_bellman_error_weight > 0.:
            # You need to replace the next memory state with the last write
            # action. See writeup for more details.
            target_observation_input = (
                self.target_policy.observation_input[0],  # o_{t+1}^buffer
                self._final_rnn_augmented_action[1]       # m_{t+1} = w_t
            )
            self.target_policy_for_policy = self.policy.get_copy(
                name_or_scope=(
                    TARGET_PREFIX + '_for_policy' + self.policy.scope_name
                ),
                observation_input=target_observation_input,
            )
            self.target_qf_for_policy = self.qf.get_copy(
                name_or_scope=(
                    TARGET_PREFIX + '_for_policy' + self.qf.scope_name
                ),
                action_input=self.target_policy_for_policy.output,
                observation_input=target_observation_input,
            )
            self.ys_for_policy = (
                self.rewards_placeholder +
                (1. - self.terminals_placeholder)
                * self.discount
                * self.target_qf_for_policy.output
            )

            action_input = (
                self.qf.action_input[0],              # a_t^buffer
                self._final_rnn_augmented_action[1],  # w_t
            )
            observation_input = (
                self.qf.observation_input[0],  # o_t^buffer
                self._final_rnn_memory_input,  # m_t
            )
            self.qf_for_policy = self.qf.get_weight_tied_copy(
                action_input=action_input,
                observation_input=observation_input,
            )
            self.bellman_error_for_policy = tf.squeeze(tf_util.mse(
                self.ys_for_policy,
                self.qf_for_policy.output
            ))

    def _init_policy_loss_and_train_ops(self):
        self.policy_surrogate_loss = - tf.reduce_mean(
            self.qf_with_action_input.output
        )
        if self._bpt_bellman_error_weight > 0.:
            self.policy_surrogate_loss += (
                self.bellman_error_for_policy * self._bpt_bellman_error_weight
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
        if not self.train_policy:
            self.train_policy_op = None

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
        feed_dict = {
            # self.qf_with_action_input.observation_input: last_obs,
            self._rnn_inputs_ph: env_obs,
            self._rnn_init_state_ph: initial_memory_obs,
            # self.policy.observation_input: last_obs,  # this is for eval to work
        }
        if self._bpt_bellman_error_weight > 0.:
            next_obs = self._split_flat_obs(batch['next_observations'])
            actions = self._split_flat_actions(batch['actions'])
            last_rewards = batch['rewards'][:, -1:]
            last_terminals = batch['terminals'][:, -1:]
            last_obs = self._get_time_step(obs, -1)
            last_next_obs = self._get_time_step(next_obs, -1)
            last_actions = self._get_time_step(actions, -1)
            feed_dict[self.qf.observation_input] = last_obs
            feed_dict[self.target_policy.observation_input] = last_next_obs
            feed_dict[self.qf.action_input] = last_actions
            feed_dict[self.rewards_placeholder] = last_rewards
            feed_dict[self.terminals_placeholder] = last_terminals
        return feed_dict


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
