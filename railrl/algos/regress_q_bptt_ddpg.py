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
from rllab.misc import special, logger


class RegressQBpttDdpg(BpttDDPG):
    """
    Train the Q function by regressing onto the oracle Q values.
    """

    def __init__(
            self,
            *args,
            oracle_qf: OracleUnrollQFunction,
            env_grad_distance_weight=0.,
            write_grad_distance_weight=0.,
            qf_grad_mse_from_one_weight=0.,
            regress_onto_values_weight=0.,
            bellman_error_weight=1.,
            **kwargs
    ):
        """

        :param args:
        :param oracle_qf:
        :param env_grad_distance_weight:
        :param write_grad_distance_weight:
        :param qf_grad_mse_from_one_weight:
        :param regress_onto_values_weight:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.oracle_qf = oracle_qf
        self.env_grad_distance_weight = env_grad_distance_weight
        self.write_grad_distance_weight = write_grad_distance_weight
        self.qf_grad_mse_from_one_weight = qf_grad_mse_from_one_weight
        self.regress_onto_values_weight = regress_onto_values_weight
        self.bellman_error_weight = bellman_error_weight

    def _init_policy_ops(self):
        super()._init_policy_ops()
        self.oracle_qf = self.oracle_qf.get_weight_tied_copy(
            action_input=self._final_rnn_augmented_action,
        )
        self.oracle_qf = self.oracle_qf.get_weight_tied_copy(
            action_input=self.qf_with_action_input.action_input,
            observation_input=self.qf_with_action_input.observation_input,
        )

    def _init_qf_loss_and_train_ops(self):
        # Compute a bunch of numbers that *could* be added to the loss.
        self.oracle_memory_grad = []
        self.oracle_env_grad = []
        self.env_grad = []
        self.memory_grad = []
        final_env_action, final_memory_action = self._rnn_outputs[-1]
        self.oracle_memory_grad = tf.gradients(self.oracle_qf.output,
                                               final_memory_action)[0]
        self.oracle_env_grad = tf.gradients(self.oracle_qf.output,
                                            final_env_action)[0]
        self.env_grad = tf.gradients(self.qf_with_action_input.output,
                                     final_env_action)[0]
        self.memory_grad = tf.gradients(self.qf_with_action_input.output,
                                        final_memory_action)[0]
        if self.oracle_memory_grad is None:
            logger.log("WARNING: Oracle memory gradients set to zero.")
            self.oracle_memory_grad = tf.zeros_like(final_memory_action)

        self.mem_grad_cosine_distance = tf_util.cosine(self.oracle_memory_grad,
                                                       self.memory_grad)
        self.env_grad_cosine_distance = tf_util.cosine(self.oracle_env_grad,
                                                       self.env_grad)
        self.mem_grad_mse = tf_util.mse(self.oracle_memory_grad,
                                        self.memory_grad, axis=1)
        self.env_grad_mse = tf_util.mse(self.oracle_env_grad,
                                        self.env_grad, axis=1)
        self.env_qf_grad_mse_from_one = tf.reduce_mean(
            (tf.abs(self.env_grad) - 1) ** 2
        )
        self.memory_qf_grad_mse_from_one = tf.reduce_mean(
            (tf.abs(self.memory_grad) - 1) ** 2
        )
        self.true_qf_mse_loss = tf.squeeze(tf_util.mse(
            self.oracle_qf.output,
            self.qf.output,
        ))

        self.qf_loss = 0.
        # The glorious if-then statement that maybe adds these numbers
        if self.qf_weight_decay > 0.:
            self.qf_loss += self.qf_weight_decay * self.Q_weights_norm
        if self.regress_onto_values_weight > 0.:
            self.qf_loss += (
                self.true_qf_mse_loss * self.regress_onto_values_weight
            )
        if self.bellman_error_weight > 0.:
            self.qf_loss += (
                self.bellman_error * self.bellman_error_weight
            )
        if self.env_grad_distance_weight > 0.:
            self.qf_loss += - (
                tf.reduce_mean(self.env_grad_cosine_distance)
            ) * self.env_grad_distance_weight
        if self.write_grad_distance_weight > 0.:
            self.qf_loss += - (
                tf.reduce_mean(self.mem_grad_cosine_distance)
            ) * self.write_grad_distance_weight
        if self.qf_grad_mse_from_one_weight > 0.:
            self.qf_loss += (
                self.env_qf_grad_mse_from_one
                + self.memory_qf_grad_mse_from_one
            ) * self.qf_grad_mse_from_one_weight
        if self.qf_loss != 0.:
            with tf.variable_scope("regress_train_qf_op"):
                self.train_qf_op = tf.train.AdamOptimizer(
                    self.qf_learning_rate
                ).minimize(
                    self.qf_loss,
                    var_list=self.qf.get_params(),
                )
        else:
            self.qf_loss = tf.zeros([0], tf.float32)
            self.train_qf_op = None

    def _qf_feed_dict(self, *args, **kwargs):
        feed_dict = super()._qf_feed_dict(*args, **kwargs)
        feed_dict.update(self._oracle_qf_feed_dict(*args, **kwargs))
        return feed_dict

    def _oracle_qf_feed_dict(self, rewards, terminals, obs, actions, next_obs,
                             target_numbers=None, times=None):
        batch_size = len(rewards)
        sequence_lengths = np.squeeze(self.env.horizon - 1 - times)
        target_one_hots = special.to_onehot_n(
            target_numbers,
            self.env.wrapped_env.action_space.flat_dim,
        )
        rest_of_obs = np.zeros(
            [
                batch_size,
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
        if hasattr(self.qf, "time_labels"):
            feed_dict[self.qf.time_labels] = times
            feed_dict[self.target_qf.time_labels] = times
        return feed_dict

    def _get_other_statistics(self):
        statistics = OrderedDict()
        for name, validation in [
            ('Valid', True),
            ('Train', False),
        ]:
            batch = self.pool.get_valid_subtrajectories(validation=validation)
            policy_feed_dict = self._policy_feed_dict_from_batch(batch)
            (
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
                oracle_qf_output,
            ) = self.sess.run(
                [
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
                    self.oracle_qf.output,
                ]
                ,
                feed_dict=policy_feed_dict
            )
            qf_feed_dict = self._qf_feed_dict_from_batch(batch)
            (
                true_qf_mse_loss,
                qf_loss,
                bellman_error,
                qf_output,
            ) = self.sess.run(
                [
                    self.true_qf_mse_loss,
                    self.qf_loss,
                    self.bellman_error,
                    self.qf.output,
                ]
                ,
                feed_dict=qf_feed_dict
            )
            stat_base_name = 'Qf{}'.format(name)
            statistics.update(
                {'{}_True_MSE_Loss'.format(stat_base_name): true_qf_mse_loss},
            )
            statistics.update(
                {'{}_BellmanError'.format(stat_base_name): bellman_error},
            )
            statistics.update(
                {'{}_Loss'.format(stat_base_name): qf_loss},
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

    def _oracle_qf_feed_dict_for_policy_from_batch(self, batch):
        all_rewards = batch['rewards']
        all_obs = batch['observations']
        all_target_numbers = batch['target_numbers']
        all_times = batch['times']

        rewards = all_rewards[:, -1]
        obs = self._split_flat_obs(self._get_time_step(all_obs, t=-1))
        target_numbers = all_target_numbers[:, -1]
        times = all_times[:, -1]
        # target_numbers = all_target_numbers.flatten()
        # times = all_times.flatten()

        batch_size = len(rewards)
        sequence_lengths = np.squeeze(self.env.horizon - 1 - times)
        target_one_hots = special.to_onehot_n(
            target_numbers,
            self.env.wrapped_env.action_space.flat_dim,
        )
        rest_of_obs = np.zeros(
            [
                batch_size,
                self.env.horizon - self._num_bptt_unrolls,
                self._env_obs_dim,
            ]
        )
        rest_of_obs[:, :, 0] = 1
        feed_dict = {
            self.oracle_qf.sequence_length_placeholder: sequence_lengths,
            self.oracle_qf.rest_of_obs_placeholder: rest_of_obs,
            self.oracle_qf.observation_input: obs,
            self.oracle_qf.target_labels: target_one_hots,
        }
        if hasattr(self.qf_with_action_input, "target_labels"):
            feed_dict[self.qf_with_action_input.target_labels] = target_one_hots
            # TODO(vitchyr): this should be the NEXT target...
            if self._bpt_bellman_error_weight > 0.:
                feed_dict[
                    self.target_qf_for_policy.target_labels] = target_one_hots
        if hasattr(self.qf_with_action_input, "time_labels"):
            feed_dict[self.qf_with_action_input.time_labels] = times
            # TODO(vitchyr): this seems hacky
            if self._bpt_bellman_error_weight > 0.:
                feed_dict[self.target_qf_for_policy.time_labels] = times + 1
        return feed_dict

    def _policy_feed_dict_from_batch(self, batch):
        policy_feed = super()._policy_feed_dict_from_batch(batch)
        policy_feed.update(self._oracle_qf_feed_dict_for_policy_from_batch(
            batch
        ))
        return policy_feed
