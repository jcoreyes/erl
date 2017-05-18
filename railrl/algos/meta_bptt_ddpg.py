from collections import OrderedDict

from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.algos.oracle_bptt_ddpg import OracleBpttDdpg
from railrl.algos.ddpg import TARGET_PREFIX, TargetUpdateMode
import tensorflow as tf

from railrl.core import tf_util
from railrl.misc.data_processing import create_stats_ordered_dict


class MetaBpttDdpg(BpttDDPG):
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
            n_steps_total=None,
    ):
        minibatch, start_indices = super()._do_training(
            n_steps_total=n_steps_total,
        )

        if self.meta_qf_output_weight > 0:
            meta_qf_ops = self._get_meta_qf_training_ops(
                n_steps_total=n_steps_total,
            )
            meta_qf_feed_dict = self._meta_qf_feed_dict_from_batch(minibatch)
            self.sess.run(meta_qf_ops, feed_dict=meta_qf_feed_dict)

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

    def _get_env_action_and_write_loss(self):
        env_action_loss, write_loss = super()._get_env_action_and_write_loss()
        if self.meta_qf_output_weight > 0:
            self.policy_meta_loss = tf.reduce_mean(
                self.meta_qf_with_action_input.output
            ) * self.meta_qf_output_weight
            env_action_loss += self.policy_meta_loss
        return env_action_loss, write_loss

    def _statistics_from_batch(self, batch) -> OrderedDict:
        statistics = super()._statistics_from_batch(batch)
        statistics.update(self._meta_qf_statistics_from_batch(batch))
        return statistics

    def _policy_statistics_from_batch(self, batch):
        policy_feed_dict = self._eval_policy_feed_dict_from_batch(batch)
        policy_stat_names, policy_ops = zip(*[
            ('PolicyMetaLoss', self.policy_meta_loss),
        ])
        values = self.sess.run(policy_ops, feed_dict=policy_feed_dict)
        statistics = super()._policy_statistics_from_batch(batch)
        for stat_name, value in zip(policy_stat_names, values):
            statistics.update(
                create_stats_ordered_dict(stat_name, value)
            )
        return statistics

    def _get_other_statistics_train_validation(self, batch, name):
        statistics = super()._get_other_statistics_train_validation(batch, name)
        policy_feed_dict = self._policy_feed_dict_from_batch(batch)
        (
            policy_meta_loss,
        ) = self.sess.run(
            [
                self.policy_meta_loss,
            ]
            ,
            feed_dict=policy_feed_dict
        )
        policy_base_stat_name = '{}Policy'.format(name)
        statistics.update(create_stats_ordered_dict(
            '{}_Meta_Loss'.format(policy_base_stat_name),
            policy_meta_loss,
        ))
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
