"""
:author: Vitchyr Pong
"""
from collections import OrderedDict
from enum import Enum

import tensorflow as tf
from typing import List

from railrl.core import tf_util
from railrl.core.neuralnet import NeuralNetwork
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import (
    split_paths,
    split_flat_product_space_into_components_n,
)
from railrl.pythonplusplus import filter_recursive
from railrl.qfunctions.nn_qfunction import NNQFunction
from railrl.tf.online_algorithm import OnlineAlgorithm
from railrl.tf.policies.nn_policy import NNPolicy
from rllab.misc.overrides import overrides
from rllab.spaces.product import Product

TARGET_PREFIX = "target_"


class TargetUpdateMode(Enum):
    SOFT = 0  # Do a soft-target update. See DDPG paper.
    HARD = 1  # Copy values over once in a while
    NONE = 2  # Don't have a target network.


class DDPG(OnlineAlgorithm):
    """
    Deep Deterministic Policy Gradient.
    """

    def __init__(
            self,
            env,
            exploration_strategy,
            policy: NNPolicy,
            qf: NNQFunction,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            qf_weight_decay=0.,
            target_update_mode=TargetUpdateMode.SOFT,
            hard_update_period=10000,
            reward_low_bellman_error_weight=0.,
            use_target_policy=True,
            **kwargs
    ):
        """
        :param env: Environment
        :param exploration_strategy: ExplorationStrategy
        :param policy: Policy that is Serializable
        :param qf: QFunctions that is Serializable
        :param qf_learning_rate: Learning rate of the qf
        :param policy_learning_rate: Learning rate of the _policy
        :param qf_weight_decay: How much to decay the weights for Q
        :param target_update_mode: How to update the target network.
        Possible options are
            - 'soft': Use a soft target update, i.e. exponential moving average
            - 'hard': Copy the values over every `hard_update_period`
        :param hard_update_period: How many epochs to between updates.
        :return:
        """
        assert isinstance(target_update_mode, TargetUpdateMode)
        super().__init__(env, policy, exploration_strategy, **kwargs)

        assert not self._batch_norm, "Batch norm not supported for DDPG."
        self._target_update_mode = target_update_mode
        self._hard_update_period = hard_update_period
        self.qf = qf
        self.qf_learning_rate = qf_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.qf_weight_decay = qf_weight_decay
        self.reward_low_bellman_error_weight = reward_low_bellman_error_weight
        self.qf_with_action_input = None
        self.target_policy = None
        self.target_qf = None
        self.target_qf_with_action_input = None
        self.use_target_policy=use_target_policy

    @overrides
    def _init_tensorflow_ops(self):
        # Initialize variables for get_copy to work
        self.sess.run(tf.global_variables_initializer())
        self.target_policy = self.policy.get_copy(
            name_or_scope=TARGET_PREFIX + self.policy.scope_name,
        )
        self.target_qf = self.qf.get_copy(
            name_or_scope=TARGET_PREFIX + self.qf.scope_name,
            action_input=self.target_policy.output,
        )
        self.qf.sess = self.sess
        self.policy.sess = self.sess
        self.target_qf.sess = self.sess
        self.target_policy.sess = self.sess
        with tf.name_scope('policy_ops'):
            self._init_policy_ops()
        with tf.name_scope('qf_ops'):
            self._init_qf_ops()
        with tf.name_scope('target_ops'):
            self._init_target_ops()
        with tf.name_scope('qf_train_ops'):
            self._init_qf_loss_and_train_ops()
        with tf.name_scope('policy_train_ops'):
            self._init_policy_loss_and_train_ops()

    def _init_qf_ops(self):
        if(not self.use_target_policy):
            self.raw_ys = tf.stop_gradient(
                self.rewards_n1 +
                (1. - self.terminals_n1)
                * self.discount
                * self.target_qf_with_action_input.output
            )
        else:
            self.raw_ys = tf.stop_gradient(
                self.rewards_n1 +
                (1. - self.terminals_n1)
                * self.discount
                * self.target_qf.output
            )

        self.raw_bellman_errors = tf.squared_difference(self.raw_ys,
                                                        self.qf.output)
        if self.reward_low_bellman_error_weight > 0.:
            self.ys = tf.stop_gradient(
                self.raw_ys
                - self.raw_bellman_errors * self.reward_low_bellman_error_weight
            )
            self.bellman_errors = tf.squared_difference(self.ys, self.qf.output)
        else:
            self.ys = self.raw_ys
            self.bellman_errors = self.raw_bellman_errors
        assert tf_util.are_shapes_compatible(
            self.rewards_n1,
            self.terminals_n1,
            self.target_qf.output,
            self.qf.output,
            self.bellman_errors,
        )
        self.bellman_error = tf.reduce_mean(self.bellman_errors)
        self.Q_weights_norm = tf.reduce_sum(
            tf.stack(
                [tf.nn.l2_loss(v)
                 for v in
                 self.qf.get_params_internal(regularizable=True)]
            ),
            name='weights_norm'
        )

    def _init_policy_ops(self):
        # To compute the surrogate loss function for the qf, it must take
        # as input the output of the _policy. See Equation (6) of "Deterministic
        # Policy Gradient Algorithms" ICML 2014.
        self.qf_with_action_input = self.qf.get_weight_tied_copy(
            observation_input=None,
            action_input=self.policy.output
        )

    def _init_target_ops(self):
        policy_vars = self.policy.get_params_internal()
        qf_vars = self.qf.get_params_internal()
        target_policy_vars = self.target_policy.get_params_internal()
        target_qf_vars = self.target_qf.get_params_internal()
        assert len(policy_vars) == len(target_policy_vars)
        assert len(qf_vars) == len(target_qf_vars)

        if self._target_update_mode == TargetUpdateMode.SOFT:
            self.update_target_policy_op = [
                tf.assign(target, (self.tau * src + (1 - self.tau) * target))
                for target, src in zip(target_policy_vars, policy_vars)]
            self.update_target_qf_op = [
                tf.assign(target, (self.tau * src + (1 - self.tau) * target))
                for target, src in zip(target_qf_vars, qf_vars)]
        elif (self._target_update_mode == TargetUpdateMode.HARD or
                      self._target_update_mode == TargetUpdateMode.NONE):
            self.update_target_policy_op = [
                tf.assign(target, src)
                for target, src in zip(target_policy_vars, policy_vars)
            ]
            self.update_target_qf_op = [
                tf.assign(target, src)
                for target, src in zip(target_qf_vars, qf_vars)
            ]
        else:
            raise RuntimeError(
                "Unknown target update mode: {}".format(
                    self._target_update_mode
                )
            )

    def _init_qf_loss_and_train_ops(self):
        self.qf_loss = (
            self.bellman_error + self.qf_weight_decay * self.Q_weights_norm
        )
        self.train_qf_op = tf.train.AdamOptimizer(
            self.qf_learning_rate
        ).minimize(
            self.qf_loss,
            var_list=self.qf.get_params_internal()
        )

    def _init_policy_loss_and_train_ops(self):
        self.policy_surrogate_loss = - tf.reduce_mean(
            self.qf_with_action_input.output,
            axis=0,
        )
        self.train_policy_op = tf.train.AdamOptimizer(
            self.policy_learning_rate
        ).minimize(
            self.policy_surrogate_loss,
            var_list=self.policy.get_params_internal(),
        )

    @overrides
    def _init_training(self):
        self._init_tensorflow_ops()
        self.sess.run(tf.global_variables_initializer())
        self.qf.reset_param_values_to_last_load()
        self.policy.reset_param_values_to_last_load()
        self.target_qf.set_param_values(self.qf.get_param_values())
        self.target_policy.set_param_values(self.policy.get_param_values())

    @overrides
    @property
    def _networks(self) -> List[NeuralNetwork]:
        return [self.policy, self.qf, self.target_policy, self.target_qf,
                self.qf_with_action_input]

    @overrides
    def _get_training_ops(
            self,
            n_steps_total=None,
    ):
        train_ops = [
            self.train_policy_op,
            self.train_qf_op,
        ]

        target_ops = []
        if self._should_update_target(n_steps_total):
            target_ops = [
                self.update_target_qf_op,
                self.update_target_policy_op,
            ]

        return filter_recursive([
            train_ops,
            target_ops,
        ])

    def _should_update_target(self, n_steps_total):
        if (self._target_update_mode == TargetUpdateMode.SOFT or
                    self._target_update_mode == TargetUpdateMode.NONE):
            return True
        elif self._target_update_mode == TargetUpdateMode.HARD:
            if n_steps_total % self._hard_update_period == 0:
                return True
        else:
            raise RuntimeError(
                "Unknown target update mode: {}".format(
                    self._target_update_mode
                )
            )
        return False

    @overrides
    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs,
                          **kwargs):
        actions = self._split_flat_actions(actions)
        obs = self._split_flat_obs(obs)
        next_obs = self._split_flat_obs(next_obs)
        qf_feed = self._qf_feed_dict(rewards,
                                     terminals,
                                     obs,
                                     actions,
                                     next_obs,
                                     **kwargs)
        policy_feed = self._policy_feed_dict(obs, **kwargs)
        # TODO(vpong): I don't think I need this
        feed = qf_feed.copy()
        feed.update(policy_feed)
        return feed

    def _split_flat_obs(self, obs):
        """
        Process vectorized version of the observations as needed
        :param obs: 1-dimensional np.ndarray
        :return: Will be given to self._qf_feed_dict and self._policy_feed_dict
        """
        if isinstance(self.env.spec.observation_space, Product):
            return split_flat_product_space_into_components_n(
                self.env.spec.observation_space,
                obs,
            )
        else:
            return obs

    def _split_flat_actions(self, actions):
        """
        Process vectorized version of the actions as needed
        :param actions: 1-dimensional np.ndarray
        :return: Will be given to self._qf_feed_dict
        """
        if isinstance(self.env.spec.action_space, Product):
            return split_flat_product_space_into_components_n(
                self.env.spec.action_space,
                actions,
            )
        else:
            return actions

    def _qf_feed_dict(self, rewards, terminals, obs, actions, next_obs,
                      **kwargs):
        """
        The output of `self._split_flat_action`.
        :return: Feed dictionary for policy training TensorFlow ops.
        """
        return {
            self.rewards_placeholder: rewards,
            self.terminals_placeholder: terminals,
            self.qf.observation_input: obs,
            self.qf.action_input: actions,
            self.target_qf.observation_input: next_obs,
            self.target_policy.observation_input: next_obs,
        }

    def _policy_feed_dict(self, obs, **kwargs):
        return {
            self.qf_with_action_input.observation_input: obs,
            self.policy.observation_input: obs,
        }

    @overrides
    def _statistics_from_paths(self, paths) -> OrderedDict:
        feed_dict = self._update_feed_dict_from_path(paths)
        stat_names, ops = zip(*self._statistic_names_and_ops())
        values = self.sess.run(ops, feed_dict=feed_dict)

        statistics = OrderedDict()
        for stat_name, value in zip(stat_names, values):
            statistics.update(
                create_stats_ordered_dict(stat_name, value)
            )

        return statistics

    def _statistic_names_and_ops(self):
        """
        :return: List of tuple (name, op). Each `op` will be evaluated. Its
        output will be saved as a statistic with name `name`.
        """
        return [
            ('PolicySurrogateLoss', self.policy_surrogate_loss),
            ('QfBellmanError', self.bellman_error),
            ('QfLoss', self.qf_loss),
            ('Ys', self.ys),
            ('PolicyOutput', self.policy.output),
            ('TargetPolicyOutput', self.target_policy.output),
            ('QfOutput', self.qf.output),
            ('TargetQfOutput', self.target_qf.output),
        ]

    def _update_feed_dict_from_path(self, paths):
        rewards, terminals, obs, actions, next_obs = split_paths(paths)
        return self._update_feed_dict(rewards, terminals, obs, actions,
                                      next_obs)

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.training_env,
            epoch=epoch,
            policy=self.policy,
            es=self.exploration_strategy,
            qf=self.qf,
        )
