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
from railrl.data_management.updatable_subtraj_replay_buffer import \
    UpdatableSubtrajReplayBuffer
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
            env_action_dim=None,
            replay_pool_size=10000,
            replay_buffer_class=SubtrajReplayBuffer,
            replay_buffer_params=None,
            bpt_bellman_error_weight=0.,
            train_policy=True,
            extra_train_period=100,
            num_extra_qf_updates=0,
            extra_qf_training_mode='none',
            qf_total_loss_tolerance=None,
            max_num_q_updates=100,
            train_qf_on_all=False,
            train_policy_on_all_qf_timesteps=False,
            write_policy_learning_rate=None,
            **kwargs
    ):
        """
        :param env: Environment
        :param exploration_strategy: ExplorationStrategy
        :param policy: Policy that is Serializable
        :param qf: QFunctions that is Serializable
        :param num_bptt_unrolls: Number of time steps to feed to the policy
        before feeding its output to the Q function.
        Must be a positive integer.
        `1` corresponds to doing just normal DDPG.
        :param bpt_bellman_error_weight:
        :param train_policy: If False, don't train the policy at all.
        :param extra_train_period: Only do extra QF training this often. Only
        used for validation mode.
        :param num_extra_qf_updates: Number of extra QF updates to do. Only used
        for fixed mode.
        :param qf_total_loss_tolerance: Stop training QF if the qf loss drops
        below this value. Only used for validation mode.
        :param max_num_q_updates: Maximum number of extra QF updates. Only
        sed for validation mode.
        :param extra_qf_training_mode: String:
        - 'none' : Don't do any extra QF training
        - 'fixed': Always do `num_extra_qf_updates` extra updates
        - 'validation': Do up to `max_num_q_updates` extra updates so long
        as validation qf loss goes down.
        :param train_qf_on_all: If True, then train the Q function on all the
        (s, a, s') tuples along the sampled sub-trajectories (rather than just
        the final tuple).
        :param train_policy_on_all_qf_timesteps: If True, then train the policy
        to maximizie all the Q(o, w, s, m) values along the sampled
        sub-trajectories (rather than just the final tuple).
        :param write_policy_learning_rate: If set, then train the write action
        part of the policy at this different learning rate. If `None`,
        the `policy_learning_rate` is used for all policy parameters. If set to
        zero, then the write action parameters aren't trained at all.
        :param kwargs: kwargs to pass onto DDPG
        """
        assert extra_qf_training_mode in [
            'none',
            'fixed',
            'validation',
        ]
        assert num_bptt_unrolls > 0
        if replay_buffer_params is None:
            replay_buffer_params = {}
        super().__init__(
            env,
            exploration_strategy,
            policy,
            qf,
            replay_pool=replay_buffer_class(
                replay_pool_size,
                env,
                num_bptt_unrolls,
                **replay_buffer_params
            ),
            **kwargs)
        self._num_bptt_unrolls = num_bptt_unrolls
        self._env_obs_dim = env_obs_dim
        self._env_action_dim = env_action_dim
        self._bpt_bellman_error_weight = bpt_bellman_error_weight
        self.train_policy = train_policy
        self.extra_qf_training_mode = extra_qf_training_mode
        self.extra_train_period = extra_train_period
        self._num_extra_qf_updates = num_extra_qf_updates
        self.qf_total_loss_tolerance = qf_total_loss_tolerance
        self.max_num_q_updates = max_num_q_updates
        self.train_qf_on_all = train_qf_on_all
        self.train_policy_on_all_qf_timesteps = train_policy_on_all_qf_timesteps
        self.write_policy_learning_rate = write_policy_learning_rate

        self._rnn_cell_scope = policy.rnn_cell_scope
        self._rnn_cell = policy.rnn_cell

        self._replay_buffer_class = replay_buffer_class
        self._replay_buffer_params = replay_buffer_params
        self._last_env_scores = []
        self.target_policy_for_policy = None
        self.target_qf_for_policy = None
        self.qf_for_policy = None

    def _sample_minibatch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return self.pool.random_subtrajectories(batch_size)

    def _do_training(
            self,
            n_steps_total=None,
    ):
        self._do_extra_qf_training(n_steps_total=n_steps_total)

        minibatch, start_indices = self._sample_minibatch()

        qf_ops = self._get_qf_training_ops(
            n_steps_total=n_steps_total,
        )
        qf_feed_dict = self._qf_feed_dict_from_batch(minibatch)
        self.sess.run(qf_ops, feed_dict=qf_feed_dict)

        policy_ops = self._get_policy_training_ops(
            n_steps_total=n_steps_total,
        )
        policy_feed_dict = self._policy_feed_dict_from_batch(minibatch)
        self.sess.run(policy_ops, feed_dict=policy_feed_dict)

        return minibatch, start_indices

    def _statistics_from_paths(self, paths) -> OrderedDict:
        eval_pool = self._replay_buffer_class(
            len(paths) * self.max_path_length,
            self.env,
            self._num_bptt_unrolls,
            **self._replay_buffer_params
        )
        for path in paths:
            eval_pool.add_trajectory(path)
        batch = eval_pool.get_all_valid_subtrajectories()
        return self._statistics_from_batch(batch)

    def _statistics_from_batch(self, batch) -> OrderedDict:
        statistics = OrderedDict()
        statistics.update(self._policy_statistics_from_batch(batch))
        statistics.update(self._qf_statistics_from_batch(batch))
        return statistics

    def _policy_statistics_from_batch(self, batch):
        policy_feed_dict = self._eval_policy_feed_dict_from_batch(batch)
        policy_stat_names, policy_ops = zip(*[
            ('PolicySurrogateLoss', self.policy_surrogate_loss),
            ('PolicyOutput', self.policy.output),
        ])
        values = self.sess.run(policy_ops, feed_dict=policy_feed_dict)
        statistics = OrderedDict()
        for stat_name, value in zip(policy_stat_names, values):
            statistics.update(
                create_stats_ordered_dict(stat_name, value)
            )
        return statistics

    def _qf_statistics_from_batch(self, batch):
        qf_feed_dict = self._eval_qf_feed_dict_from_batch(batch)
        qf_stat_names, qf_ops = zip(*[
            ('QfLoss', self.qf_loss),
            ('QfOutput', self.qf.output),
            ('TargetQfOutput', self.target_qf.output),
            ('TargetPolicyOutput', self.target_policy.output),
            ('RawYs', self.raw_ys),
            ('Ys', self.ys),
            ('QfBellmanErrors', self.bellman_errors),
            ('QfRawBellmanErrors', self.raw_bellman_errors),
        ])
        values = self.sess.run(qf_ops, feed_dict=qf_feed_dict)
        statistics = OrderedDict()
        for stat_name, value in zip(qf_stat_names, values):
            statistics.update(
                create_stats_ordered_dict(stat_name, value)
            )
        return statistics

    def _get_qf_training_ops(
            self,
            **kwargs
    ):
        return self._get_network_training_ops(
            self.train_qf_op,
            self.qf,
            self.update_target_qf_op,
            **kwargs,
        )

    def _get_network_training_ops(
            self,
            train_ops,
            network,
            target_ops,
            n_steps_total=None,
    ):
        if self._batch_norm:
            train_ops += network.batch_norm_update_stats_op

        all_target_ops = []
        if self._should_update_target(n_steps_total):
            all_target_ops = [target_ops]

        return filter_recursive([
            train_ops,
            all_target_ops,
        ])

    def _qf_feed_dict_from_batch(self, batch):
        if self.train_qf_on_all:
            flat_batch = self.subtraj_batch_to_flat_augmented_batch(batch)
        else:
            flat_batch = self.subtraj_batch_to_last_augmented_batch(batch)
        feed = self._qf_feed_dict(
            rewards=flat_batch['rewards'],
            terminals=flat_batch['terminals'],
            obs=flat_batch['obs'],
            actions=flat_batch['actions'],
            next_obs=flat_batch['next_obs'],
            target_numbers=flat_batch['target_numbers'],
            times=flat_batch['times'],
        )
        return feed

    def subtraj_batch_to_flat_augmented_batch(self, batch):
        """
        The batch is a bunch of subsequences. Flatten the subsequences so
        that they just look like normal (s, a, s') tuples.
        
        Also, the actions/observations are split into their respective 
        augmented parts.
        """
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = self._get_obs(batch)
        actions = self._get_actions(batch)
        next_obs = self._get_next_obs(batch)
        target_numbers = batch['target_numbers']
        times = batch['times']

        flat_actions = self._flatten(actions)
        flat_obs = self._flatten(obs)
        flat_next_obs = self._flatten(next_obs)
        flat_target_numbers = target_numbers.flatten()
        flat_times = times.flatten()
        flat_terminals = terminals.flatten()
        flat_rewards = rewards.flatten()

        return dict(
            rewards=flat_rewards,
            terminals=flat_terminals,
            obs=flat_obs,
            actions=flat_actions,
            next_obs=flat_next_obs,
            target_numbers=flat_target_numbers,
            times=flat_times,
        )

    def subtraj_batch_to_last_augmented_batch(self, batch):
        """
        The batch is a bunch of subsequences. Slice out the last time of each
        the subsequences so that they just look like normal (s, a, s') tuples.

        Also, the actions/observations are split into their respective
        augmented parts.
        """
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = self._get_obs(batch)
        actions = self._get_actions(batch)
        next_obs = self._get_next_obs(batch)
        target_numbers = batch['target_numbers']
        times = batch['times']

        last_actions = self._get_time_step(actions, -1)
        last_obs = self._get_time_step(obs, -1)
        last_next_obs = self._get_time_step(next_obs, -1)
        last_target_numbers = target_numbers[:, -1]
        last_times = times[:, -1]
        last_terminals = terminals[:, -1]
        last_rewards = rewards[:, -1]

        return dict(
            rewards=last_rewards,
            terminals=last_terminals,
            obs=last_obs,
            actions=last_actions,
            next_obs=last_next_obs,
            target_numbers=last_target_numbers,
            times=last_times,
        )

    def _eval_qf_feed_dict_from_batch(self, batch):
        return self._qf_feed_dict_from_batch(batch)

    """
    Extra QF Training Functions
    """

    def _do_extra_qf_training(self, n_steps_total=None, **kwargs):
        if self.extra_qf_training_mode == 'none':
            return
        elif self.extra_qf_training_mode == 'fixed':
            for _ in range(self._num_extra_qf_updates):
                minibatch, start_indices = self._sample_minibatch()
                feed_dict = self._qf_feed_dict_from_batch(minibatch)
                ops = self._get_qf_training_ops(n_steps_total=0)
                if len(ops) > 0:
                    self.sess.run(ops, feed_dict=feed_dict)
        elif self.extra_qf_training_mode == 'validation':
            if self.max_num_q_updates <= 0:
                return

            best_validation_loss = self._validation_qf_loss()
            if self._should_train_qf_extra(n_steps_total=n_steps_total):
                line_logger.print_over(
                    "{0} T:{1:3.4f} V:{2:3.4f}".format(0, 0, 0)
                )
                for i in range(self.max_num_q_updates):
                    minibatch, start_indices = self._sample_minibatch()
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
                    if validation_loss > best_validation_loss:
                        break
                    if validation_loss <= self.qf_total_loss_tolerance:
                        break
                    best_validation_loss = min(validation_loss,
                                               best_validation_loss)
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

    @property
    def qf_is_trainable(self):
        return len(self.qf.get_params()) > 0

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
        if self._num_bptt_unrolls > 1:
            self._final_rnn_memory_input = self._rnn_outputs[-2][1]
        else:
            self._final_rnn_memory_input = self._rnn_init_state_ph
        self._final_rnn_augmented_input = (
            self._rnn_inputs_unstacked[-1],
            self._final_rnn_memory_input,
        )
        self.all_env_actions = tf.concat([env_action for env_action, _ in
                                          self._rnn_outputs],
                                         axis=0)
        self.all_writes_list = [write for _, write in self._rnn_outputs]
        self.all_writes = tf.concat(self.all_writes_list, axis=0)
        self.all_actions = self.all_env_actions, self.all_writes
        self.all_mems = tf.concat(
            [self._rnn_init_state_ph] + self.all_writes_list[:-1],
            axis=0
        )
        self.all_env_obs = tf.concat(self._rnn_inputs_unstacked,
                                     axis=0)
        self.all_obs = self.all_env_obs, self.all_mems
        if self.train_policy_on_all_qf_timesteps:
            self.qf_with_action_input = self.qf.get_weight_tied_copy(
                action_input=self.all_actions,
                observation_input=self.all_obs,
            )
        else:
            self.qf_with_action_input = self.qf.get_weight_tied_copy(
                action_input=self._final_rnn_augmented_action,
                observation_input=self._final_rnn_augmented_input,
            )

        """
        Backprop the Bellman error through time, i.e. through dQ/dwrite action
        """
        if self._bpt_bellman_error_weight > 0.:
            self.next_env_obs_ph_for_policy_bpt_bellman = tf.placeholder(
                tf.float32,
                [None, self._env_obs_dim]
            )
            # You need to replace the next memory state with the last write
            # action. See writeup for more details.
            target_observation_input = (
                self.next_env_obs_ph_for_policy_bpt_bellman,  # o_{t+1}^buffer
                self._final_rnn_augmented_action[1]  # m_{t+1} = w_t
            )
            self.target_policy_for_policy = (
                self.target_policy.get_weight_tied_copy(
                    observation_input=target_observation_input,
                )
            )
            self.target_qf_for_policy = self.target_qf.get_weight_tied_copy(
                action_input=self.target_policy_for_policy.output,
                observation_input=target_observation_input,
            )
            self.ys_for_policy = (
                self.rewards_n1 +
                (1. - self.terminals_n1)
                * self.discount
                * self.target_qf_for_policy.output
            )

            self.env_action_ph_for_policy_bpt_bellman = tf.placeholder(
                tf.float32,
                [None, self._env_action_dim]
            )
            self.env_observation_ph_for_policy_bpt_bellman = tf.placeholder(
                tf.float32,
                [None, self._env_obs_dim]
            )
            action_input = (
                self.env_action_ph_for_policy_bpt_bellman,  # a_t^buffer
                self._final_rnn_augmented_action[1],  # w_t
            )
            observation_input = (
                self.env_observation_ph_for_policy_bpt_bellman,  # o_t^buffer
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
        if self.train_policy:
            self.train_policy_op = self._get_policy_train_op(
                self.policy_surrogate_loss
            )
        else:
            self.train_policy_op = None

    def _get_policy_train_op(self, loss):
        if self.write_policy_learning_rate is None:
            trainable_policy_params = self.policy.get_params_internal()
            return tf.train.AdamOptimizer(
                self.policy_learning_rate
            ).minimize(loss, var_list=trainable_policy_params)

        policy_env_params = self.policy.get_params(env_only=True)
        if self.write_policy_learning_rate == 0.:
            return tf.train.AdamOptimizer(
                self.policy_learning_rate
            ).minimize(loss, var_list=policy_env_params)
        else:
            policy_write_params = self.policy.get_params(write_only=True)
            self.train_policy_op_env = tf.train.AdamOptimizer(
                self.policy_learning_rate
            ).minimize(loss, var_list=policy_env_params)
            self.train_policy_op_write = tf.train.AdamOptimizer(
                self.write_policy_learning_rate
            ).minimize(loss, var_list=policy_write_params)
            return [
                self.train_policy_op_env,
                self.train_policy_op_write,
            ]

    def _get_policy_training_ops(self, **kwargs):
        return self._get_network_training_ops(
            self.train_policy_op,
            self.policy,
            self.update_target_policy_op,
            **kwargs
        )

    def _policy_feed_dict_from_batch(self, batch):
        obs = self._get_obs(batch)
        initial_memory_obs = self._get_time_step(obs, 0)[1]
        env_obs, _ = obs
        feed_dict = {
            self._rnn_inputs_ph: env_obs,
            self._rnn_init_state_ph: initial_memory_obs,
        }
        if self._bpt_bellman_error_weight > 0.:
            next_obs = self._get_next_obs(batch)
            actions = self._get_actions(batch)
            last_rewards = batch['rewards'][:, -1:]
            last_terminals = batch['terminals'][:, -1:]
            last_env_obs = self._get_time_step(obs, -1)[0]
            last_next_env_obs = self._get_time_step(next_obs, -1)[0]
            last_env_actions = self._get_time_step(actions, -1)[0]
            feed_dict[self.env_observation_ph_for_policy_bpt_bellman] = (
                last_env_obs
            )
            feed_dict[self.next_env_obs_ph_for_policy_bpt_bellman] = (
                last_next_env_obs
            )
            feed_dict[self.env_action_ph_for_policy_bpt_bellman] = (
                last_env_actions
            )
            feed_dict[self.rewards_placeholder] = last_rewards
            feed_dict[self.terminals_placeholder] = last_terminals
        return feed_dict

    def _eval_policy_feed_dict_from_batch(self, batch):
        feed_dict = self._policy_feed_dict_from_batch(batch)
        obs = self._get_obs(batch)
        last_obs = self._get_time_step(obs, t=-1)
        feed_dict[self.policy.observation_input] = last_obs
        return feed_dict

    """
    Miscellaneous functions
    """

    @staticmethod
    def _get_time_step(subsequences_action_or_obs, t):
        """
        Squeeze time out by only taking the one time step.

        :param subsequences_of_action_or_obs: tuple of Tensors or Tensor of
        shape [batch_size x traj_length x dim]
        :param t: The time index to slice out.
        :return: return tuple of Tensors or Tensor of shape [batch size x dim]
        """
        return map_recursive(lambda x: x[:, t, :], subsequences_action_or_obs)

    @staticmethod
    def _flatten(subsequences_of_action_or_obs):
        """
        Flatten a list of subsequences.

        :param subsequences_of_action_or_obs: tuple of Tensors or Tensor of
        shape [batch_size x traj_length x dim]
        :return: return tuple of Tensors or Tensor of shape [k x dim]
        where k = batch_size * traj_length
        """
        return map_recursive(lambda x: x.reshape(-1, x.shape[-1]),
                             subsequences_of_action_or_obs)

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

    @property
    def _networks(self):
        networks = super()._networks
        if self._bpt_bellman_error_weight > 0.:
            networks += [
                self.target_policy_for_policy,
                self.target_qf_for_policy,
                self.qf_for_policy,
            ]
        return networks

    @staticmethod
    def _get_obs(batch):
        return batch['env_obs'], batch['memories']

    @staticmethod
    def _get_next_obs(batch):
        return batch['env_obs'], batch['memories']

    @staticmethod
    def _get_actions(batch):
        return batch['env_actions'], batch['writes']
