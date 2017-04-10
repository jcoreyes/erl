"""
:author: Vitchyr Pong
"""
import numpy as np
import tensorflow as tf

from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.data_management.ocm_subtraj_replay_buffer import (
    OcmSubtrajReplayBuffer
)
from railrl.pythonplusplus import filter_recursive, print_rm_chars
from railrl.qfunctions.memory.oracle_unroll_qfunction import (
    OracleUnrollQFunction
)
from railrl.core import tf_util
from rllab.misc import special

TARGET_PREFIX = "target_"


class OracleBpttDDPG(BpttDDPG):
    """
    BpttDDPT but with an oracle QFunction.
    """
    def __init__(self, *args, **kwargs):
        kwargs['replay_buffer_class'] = OcmSubtrajReplayBuffer
        super().__init__(*args, **kwargs)

    @property
    def qf_is_trainable(self):
        return len(self.qf.get_params()) > 0

    def _init_qf_ops(self):
        if self.qf_is_trainable:
            super()._init_qf_ops()
        else:
            self.train_qf_op = None

    def _qf_feed_dict(self, rewards, terminals, obs, actions, next_obs,
                      target_numbers=None, times=None):
        indices = target_numbers[:, 0]
        target_one_hots = special.to_onehot_n(
            indices,
            self.env.wrapped_env.action_space.flat_dim,
        )
        qf_feed_dict = super()._qf_feed_dict(
            rewards=rewards,
            terminals=terminals,
            obs=obs,
            actions=actions,
            next_obs=next_obs,
        )
        qf_feed_dict[self.qf.target_labels] = target_one_hots
        qf_feed_dict[self.target_qf.target_labels] = target_one_hots
        if hasattr(self.qf, "time_labels"):
            qf_feed_dict[self.qf.time_labels] = times[:, -1]
            qf_feed_dict[self.target_qf.time_labels] = times[: -1]
        return qf_feed_dict

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
        names_and_ops = [
            ('PolicySurrogateLoss', self.policy_surrogate_loss),
            ('PolicyOutput', self.policy.output),
            ('TargetPolicyOutput', self.target_policy.output),
            ('QfOutput', self.qf_with_action_input.output),
        ]
        if self.qf_is_trainable:
            names_and_ops.append(
                ('QfLoss', self.qf_loss),
            )
        return names_and_ops


class OracleUnrollBpttDDPG(OracleBpttDDPG):
    """
    If the environment's loss is only a function of the final output,
    then you need to unroll the current policy to get the final output. This
    is what this class adds.
    """
    def __init__(self, *args, unroll_through_target_policy=False, **kwargs):
        # TODO(vitchyr): pass this in
        self.unroll_through_target_policy = unroll_through_target_policy
        super().__init__(*args, **kwargs)

    def _qf_feed_dict(self, rewards, terminals, obs, actions, next_obs,
                      target_numbers=None, times=None):
        sequence_lengths = np.squeeze(self.env.horizon - times[:, -1])
        batch_size = len(rewards)
        rest_of_obs = np.zeros(
            [
                batch_size,
                self.env.horizon - self._num_bptt_unrolls,
                self._env_obs_dim,
            ]
        )
        rest_of_obs[:, :, 0] = 1
        qf_feed_dict = super()._qf_feed_dict(
            rewards=rewards,
            terminals=terminals,
            obs=obs,
            actions=actions,
            next_obs=next_obs,
            target_numbers=target_numbers,
            times=times,
        )
        qf_feed_dict[self.qf.sequence_length_placeholder] = sequence_lengths
        qf_feed_dict[self.qf.rest_of_obs_placeholder] = rest_of_obs
        return qf_feed_dict

    def _init_policy_ops(self):
        self._rnn_inputs_ph = tf.placeholder(
            tf.float32,
            [None, self._num_bptt_unrolls, self._env_obs_dim],
            name='rnn_time_inputs',
        )
        rnn_inputs = tf.unstack(self._rnn_inputs_ph, axis=1)
        self._rnn_init_state_ph = self.policy.get_init_state_placeholder()

        # This call isn't REALLY necessary since OracleUnrollQFunction will
        # probably already make a call this scope's reuse_variable(),
        # but it's good practice to have this here.
        self._rnn_cell_scope.reuse_variables()
        self._rnn_outputs, self._rnn_final_state = tf.contrib.rnn.static_rnn(
            self._rnn_cell,
            rnn_inputs,
            initial_state=self._rnn_init_state_ph,
            dtype=tf.float32,
            scope=self._rnn_cell_scope,
        )
        self._final_rnn_output = self._rnn_outputs[-1]
        self._final_rnn_action = (
            self._final_rnn_output,
            self._rnn_final_state,
        )
        if self.unroll_through_target_policy:
            self.qf_with_action_input = self.qf.get_weight_tied_copy(
                action_input=self._final_rnn_action,
                policy=self.target_policy,
            )
        else:
            self.qf_with_action_input = self.qf.get_weight_tied_copy(
                action_input=self._final_rnn_action,
            )
        self.policy_surrogate_loss = - tf.reduce_mean(
            self.qf_with_action_input.output,
            axis=0,
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
            **kwargs
    ):
        self.qf_tolerance = qf_tolerance
        self.oracle_qf = oracle_qf
        self.max_num_q_updates = max_num_q_updates
        self.train_policy = train_policy
        self.last_qf_regression_loss = 1e10
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
        self.last_qf_regression_loss = self.sess.run(
            self.qf_loss,
            feed_dict
        )
        if self.should_train_qf(n_steps_total=n_steps_total):
            import sys
            sys.stdout.write("{0:03.3f}".format(0.0))
            for _ in range(self.max_num_q_updates):
                batch_size = min(self.pool.num_can_sample(), 128)
                minibatch = self._sample_minibatch(batch_size=batch_size)
                feed_dict = self._update_feed_dict_from_batch(minibatch)
                ops = filter_recursive([
                    self.qf_loss,
                    self.train_qf_op,
                    self.update_target_qf_op,
                ])
                self.last_qf_regression_loss = self.sess.run(ops, feed_dict=feed_dict)[0]
                print_rm_chars(11)
                sys.stdout.write("{0:03d} {1:03.3f}".format(epoch, self.last_qf_regression_loss))
                sys.stdout.flush()
                if self.last_qf_regression_loss <= self.qf_tolerance:
                    break

        super()._do_training(epoch=epoch, n_steps_total=n_steps_total,
                             n_steps_current_epoch=n_steps_current_epoch)


    def _init_policy_ops(self):
        super()._init_policy_ops()
        if not self.train_policy:
            self.train_policy_op = None

    def _init_qf_ops(self):
        # Alternatively, you could pass the action_input when creating this
        # class instance, but then you might run into serialization issues.
        self.oracle_qf = self.oracle_qf.get_weight_tied_copy(
            action_input=self.policy.output,
        )
        super()._init_qf_ops()

    def _create_qf_loss(self):
        flat_qf_output = tf.squeeze(self.qf.output, axis=1)
        return tf_util.mse(self.oracle_qf.output, flat_qf_output)

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
        if hasattr(self.qf, "time_labels"):
            feed_dict[self.qf.time_labels] = times[:, -1]
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
        if self.pool.num_can_sample(validation=True) > self.batch_size:
            batch = self.pool.get_valid_subtrajectories(validation=True)
            feed_dict = self._update_feed_dict_from_batch(batch)
            qf_validation_loss = self.sess.run(self.qf_loss, feed_dict=feed_dict)
            batch = self.pool.get_valid_subtrajectories(validation=False)
            feed_dict = self._update_feed_dict_from_batch(batch)
            qf_training_loss = self.sess.run(self.qf_loss, feed_dict=feed_dict)
            return {
                'QfValidationLoss': qf_validation_loss,
                'QfTrainingLoss': qf_training_loss,
            }
        return {}

    def _get_training_ops(
            self,
            epoch=None,
            n_steps_total=None,
            n_steps_current_epoch=None,
    ):
        """
        :return: List of ops to perform when training. If a list of list is
        provided, each list is executed in order with separate calls to
        sess.run.
        """
        ops = super()._get_training_ops()
        if not self.should_train_qf(n_steps_total):
            ops[0].remove(self.train_qf_op)
            ops[1].remove(self.update_target_qf_op)
        return ops

    def should_train_qf(self, n_steps_total):
        # return n_steps_total % 100 == 0 and self.train_qf_op is not None and self.qf_tolerance is not None
        return (
            True
            # and self.last_qf_regression_loss <= self.qf_tolerance
            and n_steps_total % 100 == 0
            and self.train_qf_op is not None
            and self.qf_tolerance is not None
        )
