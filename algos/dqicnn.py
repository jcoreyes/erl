"""
:author: Vitchyr Pong
"""
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import split_paths
from railrl.algos.online_algorithm import OnlineAlgorithm
from rllab.misc import logger
from rllab.misc import special
from rllab.misc.overrides import overrides

TARGET_PREFIX = "target_"


class DQICNN(OnlineAlgorithm):
    """
    Deep Q-learning with ICNN
    """

    def __init__(
            self,
            env,
            exploration_strategy,
            action_convex_qfunction,
            qf_learning_rate=1e-3,
            qf_weight_decay=0.,
            **kwargs
    ):
        """
        :param env: Environment
        :param exploration_strategy: ExplorationStrategy
        :param action_convex_qfunction: A NAFQFunction
        :param qf_learning_rate: Learning rate of the qf
        :param qf_weight_decay: How much to decay the weights for Q
        :return:
        """
        self.qf = action_convex_qfunction
        self.qf_learning_rate = qf_learning_rate
        self.qf_weight_decay = qf_weight_decay

        super().__init__(
            env,
            policy=None,
            exploration_strategy=exploration_strategy,
            **kwargs)

    @overrides
    def _init_tensorflow_ops(self):
        self.sess.run(tf.initialize_all_variables())
        self.next_obs_placeholder = tf.placeholder(
            tf.float32,
            shape=[None, self.observation_dim],
            name='next_obs')
        self.target_action_placeholder = tf.placeholder(
            tf.float32,
            shape=[None, self.action_dim],
            name='target_action')
        self.target_qf = self.qf.get_copy(
            name_or_scope=TARGET_PREFIX + self.qf.scope_name,
            observation_input=self.next_obs_placeholder,
            action_input=self.target_action_placeholder,
        )
        self.qf.sess = self.sess
        self.policy = self.qf.implicit_policy
        self.target_policy = self.target_qf.implicit_policy
        self.target_qf.sess = self.sess
        self._init_qf_ops()
        self._init_target_ops()
        self.sess.run(tf.initialize_all_variables())

    def _init_qf_ops(self):
        self.ys = (
            self.rewards_placeholder +
            (1. - self.terminals_placeholder) *
            self.discount * self.target_qf.output)
        self.qf_loss = tf.reduce_mean(
            tf.square(
                tf.sub(self.ys, self.qf.output)))
        self.qf_weights_norm = tf.reduce_sum(
            tf.pack(
                [tf.nn.l2_loss(v)
                 for v in
                 self.qf.get_params_internal(regularizable=True)]
            ),
            name='weights_norm'
        )
        self.qf_total_loss = (
            self.qf_loss + self.qf_weight_decay * self.qf_weights_norm)
        self.train_qf_op = tf.train.AdamOptimizer(
            self.qf_learning_rate).minimize(
            self.qf_total_loss,
            var_list=self.qf.get_params_internal())

    def _init_target_ops(self):
        qf_vars = self.qf.get_params_internal()
        target_qf_vars = self.target_qf.get_params_internal()
        assert len(qf_vars) == len(target_qf_vars)

        self.update_target_qf_op = [
            tf.assign(target, (self.tau * src + (1 - self.tau) * target))
            for target, src in zip(target_qf_vars, qf_vars)]

    @overrides
    def _init_training(self):
        self.target_qf.set_param_values(self.qf.get_param_values())

    @overrides
    def _get_training_ops(self):
        return [
            self.train_qf_op,
            self.update_target_qf_op,
        ]

    @overrides
    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        target_actions = np.vstack(
            [self.target_policy.get_action(o)[0] for o in obs]
        )
        return {
            self.rewards_placeholder: np.expand_dims(rewards, axis=1),
            self.terminals_placeholder: np.expand_dims(terminals, axis=1),
            self.qf.observation_input: obs,
            self.qf.action_input: actions,
            self.next_obs_placeholder: next_obs,
            self.target_action_placeholder: target_actions,
        }

    @overrides
    def evaluate(self, epoch, es_path_returns):
        logger.log("Collecting samples for evaluation")
        paths = self._sample_paths(epoch)
        self.log_diagnostics(paths)
        rewards, terminals, obs, actions, next_obs = split_paths(paths)
        feed_dict = self._update_feed_dict(rewards, terminals, obs, actions,
                                           next_obs)

        target_actions = feed_dict[self.target_action_placeholder]
        policy_output = [self.policy.get_action(o)[0] for o in obs]
        # Compute statistics
        (
            qf_loss,
            qf_output,
            target_qf_output,
            ys,
        ) = self.sess.run(
            [
                self.qf_loss,
                self.qf.output,
                self.target_qf.output,
                self.ys,
            ],
            feed_dict=feed_dict)
        discounted_returns = [
            special.discount_return(path["rewards"], self.discount)
            for path in paths]
        returns = [sum(path["rewards"]) for path in paths]
        rewards = np.hstack([path["rewards"] for path in paths])

        # Log statistics
        last_statistics = OrderedDict([
            ('Epoch', epoch),
            ('AverageReturn', np.mean(returns)),
            ('QfLoss', qf_loss),
        ])
        last_statistics.update(create_stats_ordered_dict('Ys', ys))
        last_statistics.update(create_stats_ordered_dict('PolicyOutput',
                                                         policy_output))
        last_statistics.update(create_stats_ordered_dict('QfOutput', qf_output))
        last_statistics.update(create_stats_ordered_dict('TargetQfOutput',
                                                         target_qf_output))
        last_statistics.update(create_stats_ordered_dict('TargetActions',
                                                         target_actions))
        last_statistics.update(create_stats_ordered_dict('Rewards', rewards))
        last_statistics.update(create_stats_ordered_dict('Returns', returns))
        last_statistics.update(create_stats_ordered_dict('DiscountedReturns',
                                                         discounted_returns))
        if len(es_path_returns) > 0:
            last_statistics.update(create_stats_ordered_dict('TrainingReturns',
                                                             es_path_returns))
        for key, value in last_statistics.items():
            logger.record_tabular(key, value)

        return self.last_statistics

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.training_env,
            epoch=epoch,
            optimizable_qfunction=self.qf,
            es=self.exploration_strategy,
        )
