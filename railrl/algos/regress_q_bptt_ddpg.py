from collections import OrderedDict

import numpy as np
import tensorflow as tf

from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.algos.ddpg import TargetUpdateMode
from railrl.core import tf_util
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.pythonplusplus import filter_recursive, line_logger
from railrl.qfunctions.memory.oracle_unroll_qfunction import (
    OracleUnrollQFunction
)
from rllab.misc import special


class RegressQBpttDdpg(BpttDDPG):
    """
    Train the Q function by regressing onto the oracle Q values.
    """

    def __init__(
            self,
            *args,
            oracle_qf: OracleUnrollQFunction,
            qf_total_loss_tolerance=None,
            max_num_q_updates=100,
            train_policy=True,
            env_grad_distance_weight=1.,
            write_grad_distance_weight=1.,
            qf_grad_mse_from_one_weight=1.,
            extra_train_period=100,
            regress_onto_values=False,
            num_extra_qf_updates=0,
            extra_qf_training_mode='none',
            validation_batch_size=None,
            **kwargs
    ):
        """

        :param args:
        :param oracle_qf:
        :param qf_total_loss_tolerance:
        :param max_num_q_updates:
        :param train_policy:
        :param env_grad_distance_weight:
        :param write_grad_distance_weight:
        :param qf_grad_mse_from_one_weight:
        :param extra_train_period:
        :param regress_onto_values:
        :param num_extra_qf_updates:
        :param extra_qf_training_mode: String:
         - 'none' : Don't do any extra QF training
         - 'fixed': Always do `num_extra_qf_updates` extra updates
         - 'validation': Do up to `max_num_q_updates` extra updates so long
         as validation qf loss goes down.
        :param kwargs:
        """
        assert extra_qf_training_mode in [
            'none',
            'fixed',
            'validation',
        ]
        super().__init__(*args, **kwargs)
        self.extra_qf_training_mode = extra_qf_training_mode
        self.qf_total_loss_tolerance = qf_total_loss_tolerance
        self.oracle_qf = oracle_qf
        self.max_num_q_updates = max_num_q_updates
        self.train_policy = train_policy
        self.env_grad_distance_weight = env_grad_distance_weight
        self.write_grad_distance_weight = write_grad_distance_weight
        self.qf_grad_mse_from_one_weight = qf_grad_mse_from_one_weight
        self.extra_train_period = extra_train_period
        self.regress_onto_values = regress_onto_values
        self._num_extra_qf_updates = num_extra_qf_updates

        self._validation_batch_size = validation_batch_size or self.batch_size

    def _do_extra_qf_training(self, n_steps_total=None, **kwargs):
        if self.extra_qf_training_mode == 'none':
            return
        elif self.extra_qf_training_mode == 'fixed':
            for _ in range(self._num_extra_qf_updates):
                minibatch = self._sample_minibatch()
                feed_dict = self._qf_update_feed_dict_from_batch(minibatch)
                ops = self._get_qf_training_ops(n_steps_total=0)
                if len(ops) > 0:
                    self.sess.run(ops, feed_dict=feed_dict)
        elif self.extra_qf_training_mode == 'validation':
            if self.max_num_q_updates <= 0:
                return

            last_validation_loss = self._validation_qf_loss()
            if self.should_train_qf_extra(n_steps_total=n_steps_total):
                line_logger.print_over(
                    "{0} T:{1:3.4f} V:{2:3.4f}".format(0, 0, 0)
                )
                for i in range(self.max_num_q_updates):
                    minibatch = self._sample_minibatch()
                    feed_dict = self._qf_update_feed_dict_from_batch(minibatch)
                    ops = [self.qf_total_loss, self.train_qf_op]
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
        feed_dict = self._qf_update_feed_dict_from_batch(batch)
        return self.sess.run(self.qf_total_loss, feed_dict=feed_dict)

    def _init_policy_ops(self):
        super()._init_policy_ops()
        if not self.train_policy:
            self.train_policy_op = None
        self.oracle_qf = self.oracle_qf.get_weight_tied_copy(
            action_input=self._final_rnn_action,
        )

    def _init_tensorflow_ops(self):
        super()._init_tensorflow_ops()

        """
        Memory gradients are w.r.t. to the LAST output of the BPTT unrolled
        network.

        For oracle QF, only the gradient w.r.t. the memory is non-zero. (The
        oracle QF trains the environment output part via the weights that are
        shared when the oracle QF unrolls the policy.) So, for the environment
        gradient, we use the ground truth environment action gradient.
        """
        self.oracle_memory_grad = []
        self.oracle_env_grad = []
        self.env_grad = []
        self.memory_grad = []
        for env_action, memory_action in self._rnn_outputs:
            self.oracle_memory_grad += tf.gradients(self.oracle_qf.output,
                                                    memory_action)
            self.oracle_env_grad += tf.gradients(self.oracle_qf.output,
                                                 env_action)
            self.env_grad += tf.gradients(self.qf_with_action_input.output,
                                          env_action)
            self.memory_grad += tf.gradients(self.qf_with_action_input.output,
                                             memory_action)
        self.oracle_memory_grad = filter_recursive(self.oracle_memory_grad)
        self.oracle_memory_grad = tf.reduce_sum(self.oracle_memory_grad, axis=0)
        self.oracle_env_grad = tf.reduce_sum(self.oracle_env_grad, axis=0)
        self.env_grad = tf.reduce_sum(self.env_grad, axis=0)
        self.memory_grad = tf.reduce_sum(self.memory_grad, axis=0)

        self.mem_grad_cosine_distance = tf_util.cosine(self.oracle_memory_grad,
                                                       self.memory_grad)
        self.env_grad_cosine_distance = tf_util.cosine(self.oracle_env_grad,
                                                       self.env_grad)
        self.mem_grad_mse = tf_util.mse(self.oracle_memory_grad,
                                        self.memory_grad, axis=1)
        self.env_grad_mse = tf_util.mse(self.oracle_env_grad,
                                        self.env_grad, axis=1)

        if self.env_grad_distance_weight > 0.:
            self.qf_total_loss += - (
                tf.reduce_mean(self.env_grad_cosine_distance) *
                self.env_grad_distance_weight
            )
        if self.write_grad_distance_weight > 0.:
            self.qf_total_loss += - (
                tf.reduce_mean(self.mem_grad_cosine_distance) *
                self.write_grad_distance_weight
            )
        self.env_qf_grad_mse_from_one = tf.reduce_mean(
            (tf.abs(self.env_grad) - 1) ** 2
        )
        self.memory_qf_grad_mse_from_one = tf.reduce_mean(
            (tf.abs(self.memory_grad) - 1) ** 2
        )
        self.qf_grad_mse_from_one = [
            self.env_qf_grad_mse_from_one,
            self.memory_qf_grad_mse_from_one,
        ]
        if self.qf_grad_mse_from_one_weight > 0.:
            self.qf_total_loss += (
                                      self.env_qf_grad_mse_from_one
                                      + self.memory_qf_grad_mse_from_one
                                  ) * self.qf_grad_mse_from_one_weight
        if self._optimize_simultaneously:
            qf_params = self.qf.get_params() + self.policy.get_params()
        else:
            qf_params = self.qf.get_params()
        with tf.variable_scope("regress_train_qf_op"):
            self.train_qf_op = tf.train.AdamOptimizer(
                self.qf_learning_rate
            ).minimize(
                self.qf_total_loss,
                var_list=qf_params,
            )

        self.sess.run(tf.global_variables_initializer())
        self.qf.reset_param_values_to_last_load()
        self.policy.reset_param_values_to_last_load()

    def _create_qf_loss(self):
        oracle_qf_output = tf.expand_dims(self.oracle_qf.output, axis=1)
        self.true_qf_mse_loss = tf.squeeze(tf_util.mse(
            oracle_qf_output,
            self.qf.output,
        ))
        if self.regress_onto_values:
            return self.true_qf_mse_loss
        else:
            return super()._create_qf_loss()

    def _qf_feed_dict(self, *args, **kwargs):
        feed_dict = super()._qf_feed_dict(*args, **kwargs)
        feed_dict.update(self._oracle_qf_feed_dict(*args, **kwargs))
        return feed_dict

    def _oracle_qf_feed_dict(self, rewards, terminals, obs, actions, next_obs,
                             target_numbers=None, times=None):
        batch_size = len(rewards)
        # sequence_lengths = np.squeeze(self.env.horizon - 1 - times[:, -1])
        # indices = target_numbers[:, 0]
        sequence_lengths = np.squeeze(self.env.horizon - 1 - times.flatten())
        indices = target_numbers.flatten()
        target_one_hots = special.to_onehot_n(
            indices,
            self.env.wrapped_env.action_space.flat_dim,
        )
        rest_of_obs = np.zeros(
            [
                # batch_size,
                len(indices),
                self.env.horizon - self._num_bptt_unrolls,
                self._env_obs_dim,
            ]
        )
        rest_of_obs[:, :, 0] = 1
        feed_dict = {
            self.oracle_qf.sequence_length_placeholder: sequence_lengths,
            self.oracle_qf.rest_of_obs_placeholder: rest_of_obs,
            self.oracle_qf.observation_input: obs,
            self.policy.observation_input: obs,
            self.oracle_qf.target_labels: target_one_hots,
        }
        if hasattr(self.qf, "target_labels"):
            feed_dict[self.qf.target_labels] = target_one_hots
            feed_dict[self.target_qf.target_labels] = target_one_hots
            last_indices = target_numbers[:, 0]
            last_target_one_hots = special.to_onehot_n(
                last_indices,
                self.env.wrapped_env.action_space.flat_dim,
            )
            feed_dict[self.qf_with_action_input.target_labels] = \
                last_target_one_hots
        if hasattr(self.qf, "time_labels"):
            feed_dict[self.qf_with_action_input.time_labels] = times[:, -1]
            # feed_dict[self.target_qf.time_labels] = times[:, -1]
            feed_dict[self.qf.time_labels] = times.flatten()
            feed_dict[self.target_qf.time_labels] = times.flatten()
        return feed_dict

    def _update_feed_dict_from_batch(self, batch):
        return self._update_feed_dict(
            rewards=batch['rewards'],
            terminals=batch['terminals'],
            obs=batch['observations'],
            actions=batch['actions'],
            next_obs=batch['next_observations'],
            target_numbers=batch['target_numbers'],
            times=batch['times'],
        )

    def _get_other_statistics(self):
        return {}
        if self.pool.num_can_sample(validation=True) < self.batch_size:
            return {}

        statistics = OrderedDict()
        for name, validation in [
            ('Valid', True),
            ('Train', False),
        ]:
            batch = self.pool.get_valid_subtrajectories(validation=validation)
            policy_feed_dict = self._policy_update_feed_dict_from_batch(batch)
            (
                true_qf_mse_loss,
                qf_loss,
                qf_total_loss,
                env_grad_distance,
                memory_grad_distance,
                env_grad_mse,
                memory_grad_mse,
                env_qf_grad_mse_from_one,
                memory_qf_grad_mse_from_one,
                env_qf_grad,
                memory_qf_grad,
                oracle_env_qf_grad,
                oracle_memory_qf_grad,
                qf_output,
                oracle_qf_output,
            ) = self.sess.run(
                [
                    self.true_qf_mse_loss,
                    self.qf_loss,
                    self.qf_total_loss,
                    self.env_grad_cosine_distance,
                    self.mem_grad_cosine_distance,
                    self.env_grad_mse,
                    self.mem_grad_mse,
                    self.env_qf_grad_mse_from_one,
                    self.memory_qf_grad_mse_from_one,
                    self.env_grad,
                    self.memory_grad,
                    self.oracle_env_grad,
                    self.oracle_memory_grad,
                    self.qf.output,
                    self.oracle_qf.output,
                ]
                ,
                feed_dict=policy_feed_dict
            )
            qf_feed_dict = self._qf_update_feed_dict_from_batch(batch)
            (
                true_qf_mse_loss,
                qf_loss,
                qf_total_loss,
                env_grad_distance,
                memory_grad_distance,
                env_grad_mse,
                memory_grad_mse,
                env_qf_grad_mse_from_one,
                memory_qf_grad_mse_from_one,
                env_qf_grad,
                memory_qf_grad,
                oracle_env_qf_grad,
                oracle_memory_qf_grad,
                qf_output,
                oracle_qf_output,
            ) = self.sess.run(
                [
                    self.true_qf_mse_loss,
                    self.qf_loss,
                    self.qf_total_loss,
                    self.env_grad_cosine_distance,
                    self.mem_grad_cosine_distance,
                    self.env_grad_mse,
                    self.mem_grad_mse,
                    self.env_qf_grad_mse_from_one,
                    self.memory_qf_grad_mse_from_one,
                    self.env_grad,
                    self.memory_grad,
                    self.oracle_env_grad,
                    self.oracle_memory_grad,
                    self.qf.output,
                    self.oracle_qf.output,
                ]
                ,
                feed_dict=qf_feed_dict
            )
            stat_base_name = 'Qf{}'.format(name)
            statistics.update(
                {'{}_True_MSE_Loss'.format(stat_base_name): true_qf_mse_loss},
            )
            statistics.update(
                {'{}_Loss'.format(stat_base_name): qf_loss},
            )
            statistics.update(
                {'{}_Total_Loss'.format(stat_base_name): qf_total_loss},
            )
            statistics.update(create_stats_ordered_dict(
                '{}_Grad_Dist_env'.format(stat_base_name),
                env_grad_distance,
            ))
            statistics.update(create_stats_ordered_dict(
                '{}_Grad_Dist_memory'.format(stat_base_name),
                memory_grad_distance
            ))
            # statistics.update(create_stats_ordered_dict(
            #     '{}_Grad_MSE_env'.format(stat_base_name),
            #     env_grad_mse,
            # ))
            # statistics.update(create_stats_ordered_dict(
            #     '{}_Grad_MSE_memory'.format(stat_base_name),
            #     memory_grad_mse
            # ))
            # statistics.update(create_stats_ordered_dict(
            #     '{}_GradMSE_from_1_env'.format(stat_base_name),
            #     env_qf_grad_mse_from_one
            # ))
            # statistics.update(create_stats_ordered_dict(
            #     '{}_GradMSE_from_1_memory'.format(stat_base_name),
            #     memory_qf_grad_mse_from_one
            # ))
            # statistics.update(create_stats_ordered_dict(
            #     '{}_QF_Grads_env'.format(stat_base_name),
            #     env_qf_grad
            # ))
            # statistics.update(create_stats_ordered_dict(
            #     '{}_QF_Grads_memory'.format(stat_base_name),
            #     memory_qf_grad
            # ))
            # statistics.update(create_stats_ordered_dict(
            #     '{}_OracleQF_Grads_env'.format(stat_base_name),
            #     oracle_env_qf_grad
            # ))
            # statistics.update(create_stats_ordered_dict(
            #     '{}_OracleQF_Grads_memory'.format(stat_base_name),
            #     oracle_memory_qf_grad
            # ))
            statistics.update(create_stats_ordered_dict(
                '{}_QfOutput'.format(stat_base_name),
                qf_output
            ))
            statistics.update(create_stats_ordered_dict(
                '{}_OracleQfOutput'.format(stat_base_name),
                oracle_qf_output
            ))
        return statistics

    def should_train_qf_extra(self, n_steps_total):
        return (
            True
            and n_steps_total % self.extra_train_period == 0
            and self.train_qf_op is not None
            and self.qf_total_loss_tolerance is not None
            and self.max_num_q_updates > 0
        )

    def _do_training(
            self,
            epoch=None,
            n_steps_total=None,
            n_steps_current_epoch=None,
    ):
        self._do_extra_qf_training(n_steps_total=n_steps_total)

        minibatch = self._sample_minibatch()

        qf_ops = self._get_qf_training_ops()
        qf_feed_dict = self._qf_update_feed_dict_from_batch(minibatch)
        self.sess.run(qf_ops, feed_dict=qf_feed_dict)

        policy_ops = self._get_policy_training_ops()
        policy_feed_dict = self._policy_update_feed_dict_from_batch(minibatch)
        self.sess.run(policy_ops, feed_dict=policy_feed_dict)

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

    def _qf_update_feed_dict_from_batch(self, batch):
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

        qf_terminals = terminals.flatten()
        qf_rewards = rewards.flatten()
        qf_obs = self._split_flat_obs(flat_obs)
        qf_actions = self._split_flat_actions(flat_actions)
        qf_next_obs = self._split_flat_obs(flat_next_obs)

        feed = self._qf_feed_dict(qf_rewards,
                                  qf_terminals,
                                  qf_obs,
                                  qf_actions,
                                  qf_next_obs,
                                  target_numbers=target_numbers,
                                  times=times
                                  )

        return feed

    def _policy_update_feed_dict_from_batch(self, batch):
        obs = batch['observations']
        policy_feed = self._policy_feed_dict(self._split_flat_obs(obs))
        return policy_feed

    def _statistics_from_paths(self, paths) -> OrderedDict:
        batch = self._batch_from_paths(paths)
        qf_feed_dict = self._qf_update_feed_dict_from_batch(batch)
        policy_feed_dict = self._policy_update_feed_dict_from_batch(batch)
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

    def _qf_statistic_names_and_ops(self):
        return [
            ('QfLoss', self.qf_loss),
            ('QfOutput', self.qf.output),
            # ('OracleQfOutput', self.oracle_qf.output),
        ]

    def _policy_statistic_names_and_ops(self):
        return [
            ('PolicySurrogateLoss', self.policy_surrogate_loss),
            ('PolicyOutput', self.policy.output),
            # ('OracleQfOutput', self.oracle_qf.output),
        ]

    def _oracle_qf_feed_dict_for_policy(
            self, rewards, terminals, obs, actions,
            next_obs, target_numbers=None,
            times=None):
        batch_size = len(rewards)
        # sequence_lengths = np.squeeze(self.env.horizon - 1 - times[:, -1])
        # indices = target_numbers[:, 0]
        sequence_lengths = np.squeeze(self.env.horizon - 1 - times.flatten())
        indices = target_numbers.flatten()
        target_one_hots = special.to_onehot_n(
            indices,
            self.env.wrapped_env.action_space.flat_dim,
        )
        rest_of_obs = np.zeros(
            [
                # batch_size,
                len(indices),
                self.env.horizon - self._num_bptt_unrolls,
                self._env_obs_dim,
            ]
        )
        rest_of_obs[:, :, 0] = 1
        feed_dict = {
            self.oracle_qf.sequence_length_placeholder: sequence_lengths,
            self.oracle_qf.rest_of_obs_placeholder: rest_of_obs,
            self.oracle_qf.observation_input: obs,
            self.policy.observation_input: obs,
            self.oracle_qf.target_labels: target_one_hots,
        }
        if hasattr(self.qf, "target_labels"):
            feed_dict[self.qf.target_labels] = target_one_hots
            feed_dict[self.target_qf.target_labels] = target_one_hots
            last_indices = target_numbers[:, 0]
            last_target_one_hots = special.to_onehot_n(
                last_indices,
                self.env.wrapped_env.action_space.flat_dim,
            )
            feed_dict[self.qf_with_action_input.target_labels] = \
                last_target_one_hots
        if hasattr(self.qf, "time_labels"):
            feed_dict[self.qf_with_action_input.time_labels] = times[:, -1]
            # feed_dict[self.target_qf.time_labels] = times[:, -1]
            feed_dict[self.qf.time_labels] = times.flatten()
            feed_dict[self.target_qf.time_labels] = times.flatten()
        return feed_dict
