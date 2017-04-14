from collections import OrderedDict

import tensorflow as tf
import numpy as np

from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.core import tf_util
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.pythonplusplus import filter_recursive, print_rm_chars
from railrl.qfunctions.memory.oracle_unroll_qfunction import \
    OracleUnrollQFunction
from rllab.misc import special


class RegressQBpttDdpg(BpttDDPG):
    """
    Train the Q function by regressing onto the oracle Q values.
    """

    def __init__(
            self,
            *args,
            oracle_qf: OracleUnrollQFunction,
            qf_tolerance=None,
            max_num_q_updates=100,
            train_policy=True,
            env_grad_distance_weight=1.,
            write_grad_distance_weight=1.,
            qf_grad_mse_from_one_weight=1.,
            extra_train_period=100,
            **kwargs
    ):
        self.qf_tolerance = qf_tolerance
        self.oracle_qf = oracle_qf
        self.max_num_q_updates = max_num_q_updates
        self.train_policy = train_policy
        self.last_qf_regression_loss = 1e10
        self.env_grad_distance_weight = env_grad_distance_weight
        self.write_grad_distance_weight = write_grad_distance_weight
        self.qf_grad_mse_from_one_weight = qf_grad_mse_from_one_weight
        self.extra_train_period = extra_train_period

        super().__init__(*args, **kwargs)

    def _do_training(
            self,
            epoch=None,
            n_steps_total=None,
            n_steps_current_epoch=None,
    ):
        batch_size = min(self.pool.num_can_sample(), 128)
        minibatch = self._sample_minibatch(batch_size=batch_size)
        feed_dict = self._update_feed_dict_from_batch(minibatch)
        self.last_qf_regression_loss = float(self.sess.run(
            self.qf_total_loss,
            feed_dict
        ))
        if self.should_train_qf_extra(n_steps_total=n_steps_total):
            import sys
            sys.stdout.write("{0:03d} {1:03.4f}".format(0, 0.0))
            i = 0
            for i in range(self.max_num_q_updates):
                batch_size = min(self.pool.num_can_sample(), 128)
                minibatch = self._sample_minibatch(batch_size=batch_size)
                feed_dict = self._update_feed_dict_from_batch(minibatch)
                ops = filter_recursive([
                    self.qf_total_loss,
                    self.train_qf_op,
                ] + self.update_target_qf_op
                )
                new_qf_regression_loss = float(
                    self.sess.run(ops, feed_dict=feed_dict)[0]
                )
                # if new_qf_regression_loss > self.last_qf_regression_loss:
                #     break
                self.last_qf_regression_loss = new_qf_regression_loss
                print_rm_chars(12)
                sys.stdout.write("{0:03d} {1:03.4f}".format(
                    i,
                    self.last_qf_regression_loss,
                ))
                sys.stdout.flush()
                if self.last_qf_regression_loss <= self.qf_tolerance:
                    break
            print_rm_chars(12)
            sys.stdout.write("{0:03d} {1:03.4f}\n".format(
                i,
                self.last_qf_regression_loss,
            ))
            sys.stdout.flush()

        super()._do_training(epoch=epoch, n_steps_total=n_steps_total,
                             n_steps_current_epoch=n_steps_current_epoch)

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
        self.oracle_grads = tf.gradients(self.oracle_qf.output,
                                         self.oracle_qf.final_actions[0])
        self.oracle_grads += tf.gradients(self.oracle_qf.output,
                                          self._final_rnn_action[1])

        self.qf_grads = tf.gradients(self.qf_with_action_input.output,
                                     list(self._final_rnn_action))

        self.grad_distance = []
        self.grad_mse = []
        # TODO(vitchyr): Have a better way of handling when the horizon = #
        # BPTT steps
        if self.oracle_grads[1] is None:
            self.oracle_grads[1] = tf.zeros_like(self.qf_grads[1])
        for oracle_grad, qf_grad in zip(self.oracle_grads, self.qf_grads):
            self.grad_distance.append(tf_util.cosine(oracle_grad, qf_grad))
            self.grad_mse.append(tf_util.mse(oracle_grad, qf_grad, axis=1))

        if self.env_grad_distance_weight >= 0.:
            self.qf_total_loss += - (
                tf.reduce_mean(self.grad_distance[0]) *
                self.env_grad_distance_weight
            )
        if self.write_grad_distance_weight >= 0.:
            self.qf_total_loss += - (
                tf.reduce_mean(self.grad_distance[1]) *
                self.write_grad_distance_weight
            )
        self.env_qf_grad_mse_from_one = tf.reduce_mean(
            (tf.abs(self.qf_grads[0]) - 1)**2
        )
        self.memory_qf_grad_mse_from_one = tf.reduce_mean(
            (tf.abs(self.qf_grads[1]) - 1)**2
        )
        self.qf_grad_mse_from_one = [
            self.env_qf_grad_mse_from_one,
            self.memory_qf_grad_mse_from_one,
        ]
        if self.qf_grad_mse_from_one_weight >= 0.:
            self.qf_total_loss += (
                self.env_qf_grad_mse_from_one
                + self.memory_qf_grad_mse_from_one
            ) * self.qf_grad_mse_from_one_weight
        with tf.variable_scope("regress_train_qf_op"):
            self.train_qf_op = tf.train.AdamOptimizer(
                self.qf_learning_rate
            ).minimize(
                self.qf_total_loss,
                var_list=self.qf.get_params_internal(),
            )

        self.sess.run(tf.global_variables_initializer())
        self.qf.reset_param_values_to_last_load()
        self.policy.reset_param_values_to_last_load()

    def _qf_feed_dict(self, *args, **kwargs):
        feed_dict = super()._qf_feed_dict(*args, **kwargs)
        feed_dict.update(self._oracle_qf_feed_dict(*args, **kwargs))
        return feed_dict

    def _oracle_qf_feed_dict(self, rewards, terminals, obs, actions, next_obs,
                             target_numbers=None, times=None):
        batch_size = len(rewards)
        sequence_lengths = np.squeeze(self.env.horizon - times[:, -1])
        indices = target_numbers[:, 0]
        target_one_hots = special.to_onehot_n(
            indices,
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
            feed_dict[self.qf.time_labels] = times[:, -1]
            feed_dict[self.target_qf.time_labels] = times[:, -1]
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

    def _statistic_names_and_ops(self):
        """
        :return: List of tuple (name, op). Each `op` will be evaluated. Its
        output will be saved as a statistic with name `name`.
        """
        return [
            ('PolicySurrogateLoss', self.policy_surrogate_loss),
            ('QfLoss', self.qf_loss),
            ('PolicyOutput', self.policy.output),
            ('QfOutput', self.qf.output),
            ('OracleQfOutput', self.oracle_qf.output),
        ]

    def _get_other_statistics(self):
        if self.pool.num_can_sample(validation=True) < self.batch_size:
            return {}

        statistics = OrderedDict()
        for name, validation in [
            ('Validation', True),
            ('Training', False),
        ]:
            batch = self.pool.get_valid_subtrajectories(validation=validation)
            feed_dict = self._update_feed_dict_from_batch(batch)
            (
                qf_loss,
                qf_total_loss,
                env_grad_distance,
                memory_grad_distance,
                env_grad_mag,
                memory_grad_mag,
                env_qf_grad_mse_from_one,
                memory_qf_grad_mse_from_one,
                env_qf_grad,
                memory_qf_grad,
            ) = self.sess.run(
                [self.qf_loss, self.qf_total_loss] + self.grad_distance +
                self.grad_mse + self.qf_grad_mse_from_one + self.qf_grads,
                feed_dict=feed_dict
            )
            stat_base_name = 'Qf_{}'.format(name)
            statistics.update(
                {'{}_MSE_Loss'.format(stat_base_name): qf_loss},
            )
            statistics.update(
                {'{}_Total_Loss'.format(stat_base_name): qf_total_loss},
            )
            statistics.update(create_stats_ordered_dict(
                '{}_Grad_Distance_env'.format(stat_base_name),
                env_grad_distance,
            ))
            statistics.update(create_stats_ordered_dict(
                '{}_Grad_Distance_memory'.format(stat_base_name),
                memory_grad_distance
            ))
            statistics.update(create_stats_ordered_dict(
                '{}_Grad_MSE_env'.format(stat_base_name),
                env_grad_mag,
            ))
            statistics.update(create_stats_ordered_dict(
                '{}_Grad_MSE_memory'.format(stat_base_name),
                memory_grad_mag
            ))
            statistics.update(create_stats_ordered_dict(
                '{}_QF_Grad_MSE_from_one_env'.format(stat_base_name),
                env_qf_grad_mse_from_one
            ))
            statistics.update(create_stats_ordered_dict(
                '{}_QF_Grad_MSE_from_one_memory'.format(stat_base_name),
                memory_qf_grad_mse_from_one
            ))
            statistics.update(create_stats_ordered_dict(
                '{}_QF_Grads_env'.format(stat_base_name),
                env_qf_grad
            ))
            statistics.update(create_stats_ordered_dict(
                '{}_QF_Grads_memory'.format(stat_base_name),
                memory_qf_grad
            ))
        return statistics

    def should_train_qf_extra(self, n_steps_total):
        return (
            True
            # and self.last_qf_regression_loss <= self.qf_tolerance
            and n_steps_total % self.extra_train_period == 0
            and self.train_qf_op is not None
            and self.qf_tolerance is not None
            and self.max_num_q_updates > 0
        )