from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.algos.ddpg import TARGET_PREFIX, TargetUpdateMode
import tensorflow as tf

from railrl.core import tf_util


class MetaBpttDdpg(BpttDDPG):
    """
    Add a meta critic: it predicts the error of the normal critic
    """
    def __init__(self, meta_qf, *args, meta_qf_learning_rate=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_qf = meta_qf
        self.meta_qf_learning_rate = meta_qf_learning_rate

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

    def _init_tensorflow_ops(self):
        super()._init_tensorflow_ops()
        self.target_meta_qf = self.meta_qf.get_copy(
            name_or_scope=TARGET_PREFIX + self.policy.scope_name,
        )
        self.meta_qf.sess = self.sess
        self.target_meta_qf.sess = self.sess
        with tf.name_scope('meta_qf_ops'):
            self._init_meta_qf_ops()
        with tf.name_scope('meta_qf_train_ops'):
            self._init_meta_qf_loss_and_train_ops()

    def _init_training(self):
        super()._init_training()
        self.meta_qf.reset_param_values_to_last_load()
        self.target_meta_qf.set_param_values(self.meta_qf.get_param_values())

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
        return self._get_training_ops(
            self.train_meta_qf_op,
            self.meta_qf,
            self.update_target_meta_qf_op,
            **kwargs,
        )

    def _meta_qf_feed_dict_from_batch(self, minibatch):
        pass

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
            self.rewards_placeholder +
            (1. - self.terminals_placeholder)
            * self.discount
            * self.target_meta_qf.output
        )
        self.meta_qf_bellman_error = tf.squeeze(
            tf_util.mse(self.meta_qf_ys, self.meta_qf.output)
        )
