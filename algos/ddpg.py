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

TARGET_PREFIX = "target_"


class DDPG(RLAlgorithm):
    """
    Deep Deterministic Policy Gradient.
    """

    def __init__(
            self,
            env,
            exploration_strategy,
            policy,
            qf,
            batch_size=32,
            n_epochs=200,
            epoch_length=1000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            soft_target_tau=1e-3,
            max_path_length=250,
            eval_samples=1000,
            scale_reward=1.,
            Q_weight_decay=1e-2,
    ):
        """
        :param env: Environment
        :param exploration_strategy: ExplorationStrategy
        :param policy: Policy that is Serializable
        :param qf: QFunctions that is Serializable
        :param replay_pool_size: Size of the replay pool
        :param batch_size: Minibatch size for training
        :param n_epochs: Number of epoch
        :param epoch_length: Number of time steps per epoch
        :param min_pool_size: Minimum size of the pool to start training.
        :param discount: Discount factor for the MDP
        :param qf_learning_rate: Learning rate of the critic
        :param policy_learning_rate: Learning rate of the actor
        :param soft_target_tau: Moving average rate. 1 = update immediately
        :param max_path_length: Maximum path length
        :param eval_samples: Number of time steps to take for evaluation.
        :param scale_reward: How much to multiply the rewards by.
        :param Q_weight_decay: How much to decay the weights for Q
        :return:
        """
        assert min_pool_size >= 2
        self.env = env
        self.actor = policy
        self.critic = qf
        self.exploration_strategy = exploration_strategy
        self.replay_pool_size = replay_pool_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.discount = discount
        self.critic_learning_rate = qf_learning_rate
        self.actor_learning_rate = policy_learning_rate
        self.tau = soft_target_tau
        self.max_path_length = max_path_length
        self.n_eval_samples = eval_samples
        self.reward_scale = scale_reward
        self.Q_weight_decay = Q_weight_decay

        self.observation_dim = self.env.observation_space.flat_dim
        self.action_dim = self.env.action_space.flat_dim
        self.rewards_placeholder = tf.placeholder(tf.float32, shape=[None, 1])
        self.terminals_placeholder = tf.placeholder(tf.float32, shape=[None, 1])
        self.pool = SimpleReplayPool(self.replay_pool_size,
                                     self.observation_dim,
                                     self.action_dim)
        self.last_statistics = None
        self.sess = tf.get_default_session() or tf.Session()
        with self.sess.as_default():
            self._init_tensorflow_ops()
        self.es_path_returns = []

    def _init_tensorflow_ops(self):
        # Initialize variables for get_copy to work
        self.sess.run(tf.initialize_all_variables())
        self.target_actor = self.actor.get_copy(
            scope_name=TARGET_PREFIX + self.actor.scope_name,
        )
        self.target_critic = self.critic.get_copy(
            scope_name=TARGET_PREFIX + self.critic.scope_name,
            action_input=self.target_actor.output
        )
        self.critic.sess = self.sess
        self.actor.sess = self.sess
        self.target_critic.sess = self.sess
        self.target_actor.sess = self.sess
        self._init_critic_ops()
        self._init_actor_ops()
        self._init_target_ops()
        self.sess.run(tf.initialize_all_variables())

    def _init_critic_ops(self):
        self.ys = (
            self.rewards_placeholder +
            (1. - self.terminals_placeholder) *
            self.discount * self.target_critic.output)
        self.critic_loss = tf.reduce_mean(
            tf.square(
                tf.sub(self.ys, self.critic.output)))
        self.Q_weights_norm = tf.reduce_sum(
            tf.pack(
                [tf.nn.l2_loss(v)
                 for v in
                 self.critic.get_params_internal(only_regularizable=True)]
            ),
            name='weights_norm'
        )
        self.critic_total_loss = (
            self.critic_loss + self.Q_weight_decay * self.Q_weights_norm)
        self.train_critic_op = tf.train.AdamOptimizer(
            self.critic_learning_rate).minimize(
            self.critic_total_loss,
            var_list=self.critic.get_params_internal())

    def _init_actor_ops(self):
        # To compute the surrogate loss function for the critic, it must take
        # as input the output of the actor. See Equation (6) of "Deterministic
        # Policy Gradient Algorithms" ICML 2014.
        self.critic_with_action_input = self.critic.get_weight_tied_copy(
            self.actor.output)
        self.actor_surrogate_loss = - tf.reduce_mean(
            self.critic_with_action_input.output)
        self.train_actor_op = tf.train.AdamOptimizer(
            self.actor_learning_rate).minimize(
            self.actor_surrogate_loss,
            var_list=self.actor.get_params_internal())

    def _init_target_ops(self):
        actor_vars = self.actor.get_params_internal()
        critic_vars = self.critic.get_params_internal()
        target_actor_vars = self.target_actor.get_params_internal()
        target_critic_vars = self.target_critic.get_params_internal()
        assert len(actor_vars) == len(target_actor_vars)
        assert len(critic_vars) == len(target_critic_vars)

        self.update_target_actor_op = [
            tf.assign(target, (self.tau * src + (1 - self.tau) * target))
            for target, src in zip(target_actor_vars, actor_vars)]
        self.update_target_critic_op = [
            tf.assign(target, (self.tau * src + (1 - self.tau) * target))
            for target, src in zip(target_critic_vars, critic_vars)]

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.actor)

    @overrides
    def train(self):
        with self.sess.as_default():
            self.target_critic.set_param_values(self.critic.get_param_values())
            self.target_actor.set_param_values(self.actor.get_param_values())
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
                                                                  self.actor)
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
                self.train_actor_op,
                self.train_critic_op,
                self.update_target_critic_op,
                self.update_target_actor_op,
            ],
            feed_dict=feed_dict)

    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        critic_feed = self._critic_feed_dict(rewards,
                                             terminals,
                                             obs,
                                             actions,
                                             next_obs)
        actor_feed = self._actor_feed_dict(obs)
        return {**critic_feed, **actor_feed}

    def _critic_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        return {
            self.rewards_placeholder: np.expand_dims(rewards, axis=1),
            self.terminals_placeholder: np.expand_dims(terminals, axis=1),
            self.critic.observations_placeholder: obs,
            self.critic.actions_placeholder: actions,
            self.target_critic.observations_placeholder: next_obs,
            self.target_actor.observations_placeholder: next_obs,
        }

    def _actor_feed_dict(self, obs):
        return {
            self.critic_with_action_input.observations_placeholder: obs,
            self.actor.observations_placeholder: obs,
        }

    def evaluate(self, epoch):
        logger.log("Collecting samples for evaluation")
        paths = parallel_sampler.sample_paths(
            policy_params=self.actor.get_param_values(),
            max_samples=self.n_eval_samples,
            max_path_length=self.max_path_length,
        )
        rewards, terminals, obs, actions, next_obs = split_paths(paths)
        feed_dict = self._update_feed_dict(rewards, terminals, obs, actions,
                                           next_obs)

        # Compute statistics
        (
            actor_loss,
            critic_loss,
            actor_outputs,
            target_actor_outputs,
            critic_outputs,
            target_critic_outputs,
            ys,
        ) = self.sess.run(
            [
                self.actor_surrogate_loss,
                self.critic_loss,
                self.actor.output,
                self.target_actor.output,
                self.critic.output,
                self.target_critic.output,
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
            ('ActorSurrogateLoss', actor_loss),
            ('ActorMeanOutput', np.mean(actor_outputs)),
            ('ActorStdOutput', np.std(actor_outputs)),
            ('TargetActorMeanOutput', np.mean(target_actor_outputs)),
            ('TargetActorStdOutput', np.std(target_actor_outputs)),
            ('CriticLoss', critic_loss),
            ('CriticMeanOutput', np.mean(critic_outputs)),
            ('CriticStdOutput', np.std(critic_outputs)),
            ('CriticMaxOutput', np.max(critic_outputs)),
            ('CriticMinOutput', np.min(critic_outputs)),
            ('TargetMeanCriticOutput', np.mean(target_critic_outputs)),
            ('TargetStdCriticOutput', np.std(target_critic_outputs)),
            ('TargetMaxCriticOutput', np.max(target_critic_outputs)),
            ('TargetMinCriticOutput', np.min(target_critic_outputs)),
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
            policy=self.actor,
            es=self.exploration_strategy,
        )
