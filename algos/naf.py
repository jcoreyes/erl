"""
:author: Vitchyr Pong
"""
import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from algos.online_algorithm import OnlineAlgorithm
from misc.rllab_util import split_paths

from misc.simple_replay_pool import SimpleReplayPool
from rllab.algos.base import RLAlgorithm
from rllab.misc import logger
from rllab.misc import special
from rllab.misc.overrides import overrides
from rllab.sampler import parallel_sampler

TARGET_PREFIX = "target_vf_of_"


class NAF(OnlineAlgorithm):
    """
    Deep Deterministic Policy Gradient.
    """

    def __init__(
            self,
            env,
            exploration_strategy,
            naf,
            qf_learning_rate=1e-3,
            Q_weight_decay=0.,
            **kwargs
    ):
        """
        :param env: Environment
        :param exploration_strategy: ExplorationStrategy
        :param naf: A NormalizedAdvantageFunction
        :param qf_learning_rate: Learning rate of the qf
        :param Q_weight_decay: How much to decay the weights for Q
        :return:
        """
        self.qf = naf
        self.qf_learning_rate = qf_learning_rate
        self.Q_weight_decay = Q_weight_decay

        policy = self.qf.get_implicit_policy()
        super().__init__(env, policy, exploration_strategy, **kwargs)

    @overrides
    def _init_tensorflow_ops(self):
        self.sess.run(tf.initialize_all_variables())
        self.policy = self.qf.get_implicit_policy()
        self.next_obs_placeholder = tf.placeholder(
            tf.float32,
            shape=[None, self.observation_dim],
            name='next_obs')
        self.target_vf = self.qf.get_implicit_value_function().get_copy(
            scope_name=TARGET_PREFIX + self.qf.scope_name,
            action_input=self.next_obs_placeholder,
        )
        self.qf.sess = self.sess
        self.target_vf.sess = self.sess
        self._init_qf_ops()
        self._init_target_ops()
        self.sess.run(tf.initialize_all_variables())

    def _init_qf_ops(self):
        self.ys = (
            self.rewards_placeholder +
            (1. - self.terminals_placeholder) *
            self.discount * self.target_vf.output)
        self.qf_loss = tf.reduce_mean(
            tf.square(
                tf.sub(self.ys, self.qf.output)))
        self.Q_weights_norm = tf.reduce_sum(
            tf.pack(
                [tf.nn.l2_loss(v)
                 for v in
                 self.qf.get_params_internal(only_regularizable=True)]
            ),
            name='weights_norm'
        )
        self.qf_total_loss = (
            self.qf_loss + self.Q_weight_decay * self.Q_weights_norm)
        self.train_qf_op = tf.train.AdamOptimizer(
            self.qf_learning_rate).minimize(
            self.qf_total_loss,
            var_list=self.qf.get_params_internal())

    def _init_target_ops(self):
        vf_vars = self.qf.vf.get_params_internal()
        target_vf_vars = self.target_vf.get_params_internal()
        assert len(vf_vars) == len(target_vf_vars)

        self.update_target_vf_op = [
            tf.assign(target, (self.tau * src + (1 - self.tau) * target))
            for target, src in zip(target_vf_vars, vf_vars)]

    @overrides
    def _init_training(self):
        self.target_vf.set_param_values(self.qf.vf.get_param_values())

    @overrides
    def _get_training_ops(self):
        return [
            self.train_qf_op,
            self.update_target_vf_op,
        ]

    @overrides
    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        return {
            self.rewards_placeholder: np.expand_dims(rewards, axis=1),
            self.terminals_placeholder: np.expand_dims(terminals, axis=1),
            self.qf.observation_input: obs,
            self.qf.action_input: actions,
            self.next_obs_placeholder: next_obs,
        }

    @overrides
    def evaluate(self, epoch, es_path_returns):
        logger.log("Collecting samples for evaluation")
        paths = parallel_sampler.sample_paths(
            policy_params=self.policy.get_param_values(),
            max_samples=self.n_eval_samples,
            max_path_length=self.max_path_length,
        )
        rewards, terminals, obs, actions, next_obs = split_paths(paths)
        feed_dict = self._update_feed_dict(rewards, terminals, obs, actions,
                                           next_obs)

        # Compute statistics
        (
            qf_loss,
            policy_output,
            qf_output,
            target_vf_output,
            ys,
        ) = self.sess.run(
            [
                self.qf_loss,
                self.policy.output,
                self.qf.output,
                self.target_vf.output,
                self.ys,
            ],
            feed_dict=feed_dict)
        average_discounted_return = np.mean(
            [special.discount_return(path["rewards"], self.discount)
             for path in paths]
        )
        returns = [sum(path["rewards"]) for path in paths]
        rewards = np.hstack([path["rewards"] for path in paths])

        # Log statistics
        last_statistics = OrderedDict([
            ('Epoch', epoch),
            ('ActorMeanOutput', np.mean(policy_output)),
            ('ActorStdOutput', np.std(policy_output)),
            ('CriticLoss', qf_loss),
            ('CriticMeanOutput', np.mean(qf_output)),
            ('CriticStdOutput', np.std(qf_output)),
            ('CriticMaxOutput', np.max(qf_output)),
            ('CriticMinOutput', np.min(qf_output)),
            ('TargetMeanCriticOutput', np.mean(target_vf_output)),
            ('TargetStdCriticOutput', np.std(target_vf_output)),
            ('TargetMaxCriticOutput', np.max(target_vf_output)),
            ('TargetMinCriticOutput', np.min(target_vf_output)),
            ('YsMean', np.mean(ys)),
            ('YsStd', np.std(ys)),
            ('YsMax', np.max(ys)),
            ('YsMin', np.min(ys)),
            ('AverageDiscountedReturn', average_discounted_return),
            ('AverageReturn', np.mean(returns)),
            ('StdReturn', np.std(returns)),
            ('MaxReturn', np.max(returns)),
            ('MinReturn', np.std(returns)),
            ('AverageRewards', np.mean(rewards)),
            ('StdRewards', np.std(rewards)),
            ('MaxRewards', np.max(rewards)),
            ('MinRewards', np.std(rewards)),
        ])
        if len(es_path_returns) > 0:
            last_statistics.update([
                ('TrainingAverageReturn', np.mean(es_path_returns)),
                ('TrainingStdReturn', np.std(es_path_returns)),
                ('TrainingMaxReturn', np.max(es_path_returns)),
                ('TrainingMinReturn', np.min(es_path_returns)),
            ])
        for key, value in last_statistics.items():
            logger.record_tabular(key, value)

        return self.last_statistics
