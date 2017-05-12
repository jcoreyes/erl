from collections import OrderedDict

from railrl.algos.oracle_bptt_ddpg import OracleBpttDdpg
from railrl.algos.ddpg import TARGET_PREFIX, TargetUpdateMode
import tensorflow as tf

from railrl.core import tf_util
from railrl.misc.data_processing import create_stats_ordered_dict


class MetaBpttDdpg(OracleBpttDdpg):
    """
    Add a meta critic: it predicts the error of the normal critic
    """

    def __init__(
            self,
            *args,
            meta_qf=None,
            meta_qf_learning_rate=1e-4,
            meta_qf_output_weight=10,
            qf_output_weight=1,
            train_meta_qf_on_all=True,
            **kwargs
    ):
        """
        :param args: args to pass to OracleBpttDdpg
        :param meta_qf: Meta QF to predict Bellman Error
        :param meta_qf_learning_rate: Learning rate for meta QF
        :param meta_qf_output_weight: How much to scale the output of the
        meta QF output for the policy
        :param qf_output_weight: How much to scale the output of the
        normal QF output for the policy
        :param kwargs: kwargs to pass to OracleBpttDdpg
        """
        super().__init__(*args, **kwargs)
        self.meta_qf = meta_qf
        self.meta_qf_learning_rate = meta_qf_learning_rate
        self.meta_qf_output_weight = meta_qf_output_weight
        self.qf_output_weight = qf_output_weight
        self.train_meta_qf_on_all = train_meta_qf_on_all

    def _do_training(
            self,
            epoch=None,
            n_steps_total=None,
            n_steps_current_epoch=None,
    ):
        self._do_extra_qf_training(n_steps_total=n_steps_total)

        minibatch = self._sample_minibatch()

        qf_ops = self._get_qf_training_ops(
            epoch=epoch,
            n_steps_total=n_steps_total,
            n_steps_current_epoch=n_steps_current_epoch,
        )
        qf_feed_dict = self._qf_feed_dict_from_batch(minibatch)
        self.sess.run(qf_ops, feed_dict=qf_feed_dict)

        if self.meta_qf_output_weight > 0:
            meta_qf_ops = self._get_meta_qf_training_ops(
                epoch=epoch,
                n_steps_total=n_steps_total,
                n_steps_current_epoch=n_steps_current_epoch,
            )
            meta_qf_feed_dict = self._meta_qf_feed_dict_from_batch(minibatch)
            self.sess.run(meta_qf_ops, feed_dict=meta_qf_feed_dict)

        policy_ops = self._get_policy_training_ops(
            epoch=epoch,
            n_steps_total=n_steps_total,
            n_steps_current_epoch=n_steps_current_epoch,
        )
        policy_feed_dict = self._policy_feed_dict_from_batch(minibatch)
        self.sess.run(policy_ops, feed_dict=policy_feed_dict)

    def _init_training(self):
        super()._init_training()
        self.meta_qf.reset_param_values_to_last_load()
        self.target_meta_qf.set_param_values(self.meta_qf.get_param_values())

    def _init_tensorflow_ops(self):
        # Initialize variables for get_copy to work
        self.sess.run(tf.global_variables_initializer())
        self.meta_qf = self.meta_qf.get_copy(
            name_or_scope="copied_" + self.meta_qf.scope_name,
            action_input=self.qf.action_input,
            observation_input=self.qf.observation_input,
        )
        self.target_meta_qf = self.meta_qf.get_copy(
            name_or_scope=TARGET_PREFIX + self.meta_qf.scope_name,
            action_input=self.meta_qf.action_input,
            observation_input=self.meta_qf.observation_input,
        )
        super()._init_tensorflow_ops()
        self.meta_qf.sess = self.sess
        self.target_meta_qf.sess = self.sess
        with tf.name_scope('meta_qf_ops'):
            self._init_meta_qf_ops()
        with tf.name_scope('meta_qf_train_ops'):
            self._init_meta_qf_loss_and_train_ops()

    def _init_target_ops(self):
        super()._init_target_ops()
        meta_qf_vars = self.meta_qf.get_params_internal()
        target_meta_qf_vars = self.target_meta_qf.get_params_internal()
        assert len(meta_qf_vars) == len(target_meta_qf_vars)

        if self._target_update_mode == TargetUpdateMode.SOFT:
            self.update_target_meta_qf_op = [
                tf.assign(target, (self.tau * src + (1 - self.tau) * target))
                for target, src in zip(target_meta_qf_vars, meta_qf_vars)]
        elif (self._target_update_mode == TargetUpdateMode.HARD or
                self._target_update_mode == TargetUpdateMode.NONE):
            self.update_target_meta_qf_op = [
                tf.assign(target, src)
                for target, src in zip(target_meta_qf_vars, meta_qf_vars)
            ]
        else:
            raise RuntimeError(
                "Unknown target update mode: {}".format(
                    self._target_update_mode
                )
            )

    def _get_meta_qf_training_ops(self, **kwargs):
        return self._get_network_training_ops(
            self.train_meta_qf_op,
            self.meta_qf,
            self.update_target_meta_qf_op,
            **kwargs,
        )

    def _meta_qf_feed_dict_from_batch(self, batch):
        if self.train_qf_on_all:
            flat_batch = self.subtraj_batch_to_flat_augmented_batch(batch)
        else:
            flat_batch = self.subtraj_batch_to_last_augmented_batch(batch)
        feed_dict = self._qf_feed_dict(
            rewards=flat_batch['rewards'],
            terminals=flat_batch['terminals'],
            obs=flat_batch['obs'],
            actions=flat_batch['actions'],
            next_obs=flat_batch['next_obs'],
            target_numbers=flat_batch['target_numbers'],
            times=flat_batch['times'],
        )
        flat_target_labels = flat_batch['target_numbers']
        flat_times = flat_batch['times']
        feed_dict.update({
            self.meta_qf.target_labels: flat_target_labels,
            self.meta_qf.time_labels: flat_times,
            self.target_meta_qf.target_labels: flat_target_labels,
            self.target_meta_qf.time_labels: flat_times,
        })
        return feed_dict

    def _oracle_qf_feed_dict_for_policy_from_batch(self, batch):
        feed_dict = super()._oracle_qf_feed_dict_for_policy_from_batch(batch)
        # (
        #     last_rewards,
        #     last_obs,
        #     episode_length_left,
        #     target_one_hots,
        #     last_times,
        #     rest_of_obs,
        # ) = self._get_last_time_step_from_batch(batch)
        flat_batch = self.subtraj_batch_to_flat_augmented_batch(batch)
        last_times = flat_batch['times']
        target_labels = flat_batch['target_numbers']
        feed_dict.update({
            self.meta_qf.target_labels: target_labels,
            self.meta_qf.time_labels: last_times,
            self.target_meta_qf.target_labels: target_labels,
            self.target_meta_qf.time_labels: last_times,
        })
        return feed_dict

    def _init_meta_qf_loss_and_train_ops(self):
        self.meta_qf_loss = self.meta_qf_bellman_error
        self.train_meta_qf_op = tf.train.AdamOptimizer(
            self.meta_qf_learning_rate
        ).minimize(
            self.meta_qf_loss,
            var_list=self.meta_qf.get_params_internal()
        )

    def _init_meta_qf_ops(self):
        self.meta_qf_ys = (
            self.bellman_errors +
            (1. - self.terminals_n1)
            * self.discount
            * self.target_meta_qf.output
        )
        self.meta_qf_bellman_errors = tf.squared_difference(
            self.meta_qf_ys, self.meta_qf.output
        )
        assert tf_util.are_shapes_compatible(
            self.bellman_errors,
            self.terminals_n1,
            self.target_meta_qf.output,
            self.meta_qf.output,
            self.meta_qf_bellman_errors,
        )
        self.meta_qf_bellman_error = tf.reduce_mean(
            self.meta_qf_bellman_errors
        )

    def _init_policy_ops(self):
        super()._init_policy_ops()
        self.meta_qf_with_action_input = self.meta_qf.get_weight_tied_copy(
            action_input=self.qf_with_action_input.action_input,
            observation_input=self.qf_with_action_input.observation_input,
        )

    def _init_policy_loss_and_train_ops(self):
        self.policy_surrogate_loss = - tf.reduce_mean(
            self.qf_with_action_input.output
        ) * self.qf_output_weight
        if self.meta_qf_output_weight > 0:
            self.policy_surrogate_loss += tf.reduce_mean(
                self.meta_qf_with_action_input.output
            ) * self.meta_qf_output_weight
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

    def _statistics_from_batch(self, batch) -> OrderedDict:
        statistics = super()._statistics_from_batch(batch)
        statistics.update(self._meta_qf_statistics_from_batch(batch))
        return statistics

    def _meta_qf_statistics_from_batch(self, batch):
        meta_qf_feed_dict = self._eval_meta_qf_feed_dict_from_batch(batch)
        meta_qf_stat_names, meta_qf_ops = zip(*[
            ('MetaQfLoss', self.meta_qf_loss),
            ('MetaQfOutput', self.meta_qf.output),
            ('MetaQfBellmanErrors', self.meta_qf_bellman_errors),
        ])
        values = self.sess.run(meta_qf_ops, feed_dict=meta_qf_feed_dict)
        statistics = OrderedDict()
        for stat_name, value in zip(meta_qf_stat_names, values):
            statistics.update(
                create_stats_ordered_dict(stat_name, value)
            )
        return statistics

    def _eval_meta_qf_feed_dict_from_batch(self, batch):
        return self._meta_qf_feed_dict_from_batch(batch)

    @property
    def _networks(self):
        return super()._networks + [
            self.meta_qf,
            self.target_meta_qf,
            self.meta_qf_with_action_input,
        ]
