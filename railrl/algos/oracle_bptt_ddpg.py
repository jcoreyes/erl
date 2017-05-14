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
from rllab.misc import logger


class OracleBpttDdpg(BpttDDPG):
    """
    Have an oracle Q function, either for debugging or to get ground truth.
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
            use_oracle_qf=False,
            unroll_through_target_policy=False,
            **kwargs
    ):
        """

        :param args:
        :param oracle_qf:
        :param env_grad_distance_weight:
        :param write_grad_distance_weight:
        :param qf_grad_mse_from_one_weight:
        :param regress_onto_values_weight:
        :param use_oracle_qf: If True, replace the qf with the oracle qf
        :param unroll_through_target_policy: If True, unroll the oracle qf
        through the target policy.
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.oracle_qf = oracle_qf
        self.env_grad_distance_weight = env_grad_distance_weight
        self.write_grad_distance_weight = write_grad_distance_weight
        self.qf_grad_mse_from_one_weight = qf_grad_mse_from_one_weight
        self.regress_onto_values_weight = regress_onto_values_weight
        self.bellman_error_weight = bellman_error_weight
        self.use_oracle_qf = use_oracle_qf
        self.unroll_through_target_policy = unroll_through_target_policy

    def _init_policy_ops(self):
        super()._init_policy_ops()
        action_input = self.qf_with_action_input.action_input
        obs_input = self.qf_with_action_input.observation_input
        self.oracle_qf = self.oracle_qf.get_weight_tied_copy(
            action_input=action_input,
            observation_input=obs_input,
        )
        if self.use_oracle_qf:
            if self.unroll_through_target_policy:
                self.qf_with_action_input = self.oracle_qf.get_weight_tied_copy(
                    action_input=action_input,
                    observation_input=obs_input,
                    policy=self.target_policy,
                )
            else:
                self.qf_with_action_input = self.oracle_qf.get_weight_tied_copy(
                    action_input=action_input,
                    observation_input=obs_input,
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
        if self.memory_grad is None:
            logger.log("WARNING: Memory gradients set to zero.")
            self.memory_grad = tf.zeros_like(final_memory_action)

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
        if self.qf_loss != 0. and self.qf_is_trainable:
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

    def _qf_feed_dict_from_batch(self, batch):
        feed_dict = super()._qf_feed_dict_from_batch(batch)
        feed_dict.update(self._oracle_qf_feed_dict_from_batch(batch))
        return feed_dict

    def _oracle_qf_feed_dict_from_batch(self, batch):
        all_flat_batch = self.subtraj_batch_to_flat_augmented_batch(batch)
        if self.train_qf_on_all:
            flat_batch = all_flat_batch
        else:
            flat_batch = self.subtraj_batch_to_last_augmented_batch(batch)
        times = flat_batch['times']
        batch_size = len(times)
        sequence_lengths = self.env.horizon - 1 - times
        # TODO(vitchyr): BUG this gets more complicated with the flags. I
        # should make the environment generate the rest of the observations.
        rest_of_obs = np.zeros(
            [
                batch_size,
                self.env.horizon - self._num_bptt_unrolls,
                self._env_obs_dim,
            ]
        )
        all_env_obs, all_memories = all_flat_batch['obs']
        obs = flat_batch['obs']
        target_numbers = flat_batch['target_numbers']
        # TODO: oracle_qf shouldn't depend on policy.
        split_all_env_obs = all_env_obs.reshape(
            (-1, self._num_bptt_unrolls, all_env_obs.shape[-1])
        )
        feed_dict = {
            self.oracle_qf.sequence_length_placeholder: sequence_lengths,
            self.oracle_qf.rest_of_obs_placeholder: rest_of_obs,
            self.oracle_qf.observation_input: obs,
            self.policy.observation_input: obs,
            self._rnn_inputs_ph: split_all_env_obs,
            self._rnn_init_state_ph: all_memories[::self._num_bptt_unrolls, :],
            self.oracle_qf.target_labels: target_numbers,
        }
        if hasattr(self.qf, "target_labels"):
            feed_dict[self.qf.target_labels] = target_numbers
            feed_dict[self.target_qf.target_labels] = target_numbers
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
                bellman_errors,
                qf_output,
            ) = self.sess.run(
                [
                    self.true_qf_mse_loss,
                    self.qf_loss,
                    self.bellman_errors,
                    self.qf.output,
                ]
                ,
                feed_dict=qf_feed_dict
            )
            stat_base_name = 'Qf{}'.format(name)
            statistics.update(create_stats_ordered_dict(
                '{}_True_MSE_Loss'.format(stat_base_name),
                true_qf_mse_loss,
            ))
            statistics.update(create_stats_ordered_dict(
                '{}_BellmanError'.format(stat_base_name),
                bellman_errors,
            ))
            statistics.update(create_stats_ordered_dict(
                '{}_Loss'.format(stat_base_name),
                qf_loss,
            ))
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
        if self.train_policy_on_all_qf_timesteps:
            flat_batch = self.subtraj_batch_to_flat_augmented_batch(batch)
        else:
            flat_batch = self.subtraj_batch_to_last_augmented_batch(batch)
        target_numbers = flat_batch['target_numbers']
        times = flat_batch['times']
        obs = flat_batch['obs']
        episode_length_left = self.env.horizon - 1 - times
        rest_of_obs = np.zeros(
            [
                len(target_numbers),
                self.env.horizon - self._num_bptt_unrolls,
                self._env_obs_dim,
            ]
        )
        feed_dict = {
            self.oracle_qf.sequence_length_placeholder: episode_length_left,
            self.oracle_qf.rest_of_obs_placeholder: rest_of_obs,
            # It's better to separate them so that duplicate entries can be
            # eliminated by TensorFlow
            self.oracle_qf.observation_input[0]: obs[0],
            self.oracle_qf.observation_input[1]: obs[1],
            self.oracle_qf.target_labels: target_numbers,
        }

        if hasattr(self.qf_with_action_input, "target_labels"):
            feed_dict[self.qf_with_action_input.target_labels] = target_numbers
        if hasattr(self.qf_with_action_input, "time_labels"):
            feed_dict[self.qf_with_action_input.time_labels] = times
        if self.target_qf_for_policy is not None:
            if (hasattr(self.target_qf_for_policy, "target_numbers") and
                        self._bpt_bellman_error_weight > 0.):
                # TODO(vitchyr): this should be the NEXT target...
                feed_dict[self.target_qf_for_policy.target_numbers] = (
                    target_numbers
                )
            if (hasattr(self.target_qf_for_policy, "time_labels") and
                     self._bpt_bellman_error_weight > 0.):
                # TODO(vitchyr): this seems hacky
                feed_dict[self.target_qf_for_policy.time_labels] = times + 1
        return feed_dict

    def _policy_feed_dict_from_batch(self, batch):
        policy_feed = super()._policy_feed_dict_from_batch(batch)
        policy_feed.update(self._oracle_qf_feed_dict_for_policy_from_batch(
            batch
        ))
        return policy_feed

    @property
    def _networks(self):
        return super()._networks + [
            self.oracle_qf,
        ]
