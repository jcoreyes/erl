"""
:author: Vitchyr Pong
"""
import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from misc.rllab_util import split_paths

from misc.simple_replay_pool import SimpleReplayPool
from rllab.algos.base import RLAlgorithm
from rllab.misc import logger
from rllab.misc import special
from rllab.misc.overrides import overrides
from rllab.sampler import parallel_sampler

TARGET_PREFIX = "target_vf_of_"


class NAF(RLAlgorithm):
    """
    Deep Deterministic Policy Gradient.
    """

    def __init__(
            self,
            env,
            exploration_strategy,
            qf,
            batch_size=64,
            n_epochs=1000,
            epoch_length=1000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            discount=0.99,
            qf_learning_rate=1e-3,
            soft_target_tau=1e-2,
            max_path_length=1000,
            eval_samples=1000,
            scale_reward=1.,
            Q_weight_decay=0.,
    ):
        """
        :param env: Environment
        :param exploration_strategy: ExplorationStrategy
        :param qf: QFunctions that is Serializable
        :param replay_pool_size: Size of the replay pool
        :param batch_size: Minibatch size for training
        :param n_epochs: Number of epoch
        :param epoch_length: Number of time steps per epoch
        :param min_pool_size: Minimum size of the pool to start training.
        :param discount: Discount factor for the MDP
        :param qf_learning_rate: Learning rate of the qf
        :param soft_target_tau: Moving average rate. 1 = update immediately
        :param max_path_length: Maximum path length
        :param eval_samples: Number of time steps to take for evaluation.
        :param scale_reward: How much to multiply the rewards by.
        :param Q_weight_decay: How much to decay the weights for Q
        :return:
        """
        assert min_pool_size >= 2
        self.env = env
        self.qf = qf
        self.exploration_strategy = exploration_strategy
        self.replay_pool_size = replay_pool_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.discount = discount
        self.qf_learning_rate = qf_learning_rate
        self.tau = soft_target_tau
        self.max_path_length = max_path_length
        self.n_eval_samples = eval_samples
        self.reward_scale = scale_reward
        self.Q_weight_decay = Q_weight_decay

        self.observation_dim = self.env.observation_space.flat_dim
        self.action_dim = self.env.action_space.flat_dim
        self.rewards_placeholder = tf.placeholder(tf.float32,
                                                  shape=[None, 1],
                                                  name='rewards')
        self.terminals_placeholder = tf.placeholder(tf.float32,
                                                    shape=[None, 1],
                                                    name='terminals')
        self.next_obs_placeholder = tf.placeholder(
            tf.float32,
            shape=[None, self.observation_dim],
            name='next_obs')
        self.pool = SimpleReplayPool(self.replay_pool_size,
                                     self.observation_dim,
                                     self.action_dim)
        self.last_statistics = None
        self.sess = tf.get_default_session() or tf.Session()
        with self.sess.as_default():
            self._init_tensorflow_ops()
        self.es_path_returns = []

    def _init_tensorflow_ops(self):
        self.sess.run(tf.initialize_all_variables())
        self.policy = self.qf.get_implicit_policy()
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

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)

    @overrides
    def train(self):
        with self.sess.as_default():
            self.target_vf.set_param_values(self.qf.vf.get_param_values())
            self.start_worker()

            observation = self.env.reset()
            self.exploration_strategy.reset()
            itr = 0
            path_length = 0
            path_return = 0
            for epoch in range(self.n_epochs):
                logger.push_prefix('Epoch #%d | ' % epoch)
                logger.log("Training started")
                start_time = time.time()
                for _ in range(self.epoch_length):
                    action = self.exploration_strategy.get_action(itr,
                                                                  observation,
                                                                  self.policy)
                    next_ob, raw_reward, terminal, _ = self.env.step(action)
                    reward = raw_reward * self.reward_scale
                    path_length += 1
                    path_return += reward

                    self.pool.add_sample(observation,
                                         action,
                                         reward,
                                         terminal,
                                         False)
                    if terminal or path_length >= self.max_path_length:
                        self.pool.add_sample(next_ob,
                                             np.zeros_like(action),
                                             np.zeros_like(reward),
                                             np.zeros_like(terminal),
                                             True)
                        observation = self.env.reset()
                        self.exploration_strategy.reset()
                        self.es_path_returns.append(path_return)
                        path_length = 0
                        path_return = 0
                    else:
                        observation = next_ob

                    if self.pool.size >= self.min_pool_size:
                        self.do_training()
                    itr += 1

                logger.log("Training finished. Time: {0}".format(time.time() -
                                                                 start_time))
                if self.pool.size >= self.min_pool_size:
                    start_time = time.time()
                    self.evaluate(epoch)
                    params = self.get_epoch_snapshot(epoch)
                    logger.log(
                        "Eval time: {0}".format(time.time() - start_time))
                    logger.save_itr_params(epoch, params)
                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()
            self.env.terminate()
            return self.last_statistics

    def do_training(self):
        minibatch = self.pool.random_batch(self.batch_size)
        sampled_obs = minibatch['observations']
        sampled_terminals = minibatch['terminals']
        sampled_actions = minibatch['actions']
        sampled_rewards = minibatch['rewards']
        sampled_next_obs = minibatch['next_observations']

        feed_dict = self._update_feed_dict(sampled_rewards,
                                           sampled_terminals,
                                           sampled_obs,
                                           sampled_actions,
                                           sampled_next_obs)
        self.sess.run(
            [
                self.train_qf_op,
                self.update_target_vf_op,
            ],
            feed_dict=feed_dict)

    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        return {
            self.rewards_placeholder: np.expand_dims(rewards, axis=1),
            self.terminals_placeholder: np.expand_dims(terminals, axis=1),
            self.qf.observation_input: obs,
            self.qf.action_input: actions,
            self.next_obs_placeholder: next_obs,
        }

    def evaluate(self, epoch):
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
        self.last_statistics = OrderedDict([
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
        if len(self.es_path_returns) > 0:
            self.last_statistics.update([
                ('TrainingAverageReturn', np.mean(self.es_path_returns)),
                ('TrainingStdReturn', np.std(self.es_path_returns)),
                ('TrainingMaxReturn', np.max(self.es_path_returns)),
                ('TrainingMinReturn', np.min(self.es_path_returns)),
            ])
        for key, value in self.last_statistics.items():
            logger.record_tabular(key, value)

        self.es_path_returns = []

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.env,
            epoch=epoch,
            policy=self.policy,
            es=self.exploration_strategy,
        )
