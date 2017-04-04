"""
:author: Vitchyr Pong
"""
import abc
from collections import OrderedDict
import pickle
import time
from contextlib import contextmanager
from typing import Iterable

import numpy as np
import tensorflow as tf

from railrl.data_management.replay_buffer import ReplayBuffer
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.policies.nn_policy import NNPolicy
from railrl.core.neuralnet import NeuralNetwork
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from rllab.algos.base import RLAlgorithm
from rllab.misc import logger, special
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler


class OnlineAlgorithm(RLAlgorithm):
    """
    Online learning algorithm.
    """

    def __init__(
            self,
            env,
            policy: NNPolicy,
            exploration_strategy,
            batch_size=64,
            n_epochs=1000,
            epoch_length=10000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            discount=0.99,
            soft_target_tau=1e-2,
            max_path_length=1000,
            eval_samples=10000,
            scale_reward=1.,
            render=False,
            n_updates_per_time_step=1,
            batch_norm_config=None,
            replay_pool: ReplayBuffer = None,
            allow_gpu_growth=True,
    ):
        """
        :param env: Environment
        :param exploration_strategy: ExplorationStrategy
        :param policy: A Policy
        :param replay_pool_size: Size of the replay pool
        :param batch_size: Minibatch size for training
        :param n_epochs: Number of epoch
        :param epoch_length: Number of time steps per epoch
        :param min_pool_size: Minimum size of the pool to start training.
        :param discount: Discount factor for the MDP
        :param soft_target_tau: Moving average rate. 1 = update immediately
        :param max_path_length: Maximum path length
        :param eval_samples: Number of time steps to take for evaluation.
        :param scale_reward: How much to multiply the rewards by.
        :param render: Boolean. If True, render the environment.
        :param n_updates_per_time_step: How many SGD steps to take per time
        step.
        :param batch_norm_config: Config for batch_norm. If set, batch_norm
        is enabled.
        :param replay_pool: Replay pool class
        :param allow_gpu_growth: Allow the GPU to grow. If True, TensorFlow
        won't pre-allocate memory, but this will be a bit slower.

        http://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
        :return:
        """
        assert min_pool_size >= batch_size
        # Have two separate env's to make sure that the training and eval
        # envs don't affect one another.
        self.training_env = env
        self.env = pickle.loads(pickle.dumps(self.training_env))
        self.policy = policy
        self.exploration_strategy = exploration_strategy
        self.replay_pool_size = replay_pool_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.discount = discount
        self.tau = soft_target_tau
        self.max_path_length = max_path_length
        self.n_eval_samples = eval_samples
        self.scale_reward = scale_reward
        self.render = render
        self.n_updates_per_time_step = n_updates_per_time_step
        self._batch_norm = batch_norm_config is not None
        self._batch_norm_config = batch_norm_config

        self.observation_dim = self.training_env.observation_space.flat_dim
        self.action_dim = self.training_env.action_space.flat_dim
        self.rewards_placeholder = tf.placeholder(tf.float32,
                                                  shape=[None, 1],
                                                  name='rewards')
        self.terminals_placeholder = tf.placeholder(tf.float32,
                                                    shape=[None, 1],
                                                    name='terminals')
        self.pool = replay_pool or EnvReplayBuffer(
            self.replay_pool_size,
            self.env,
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = allow_gpu_growth
        self.sess = tf.Session(config=config)
        self.es_path_returns = []

        self.eval_sampler = BatchSampler(self)
        self.scope = None  # Necessary for BatchSampler
        self.whole_paths = True  # Also for BatchSampler
        self._last_average_returns = None

    def _start_worker(self):
        self.eval_sampler.start_worker()

    def _shutdown_worker(self):
        self.eval_sampler.shutdown_worker()

    def _sample_paths(self, epoch):
        """
        Returns flattened paths.

        :param epoch: Epoch number
        :return: Dictionary with these keys:
            observations: np.ndarray, shape BATCH_SIZE x flat observation dim
            actions: np.ndarray, shape BATCH_SIZE x flat action dim
            rewards: np.ndarray, shape BATCH_SIZE
            terminals: np.ndarray, shape BATCH_SIZE
            agent_infos: unsure
            env_infos: unsure
        """
        # Sampler uses self.batch_size to figure out how many samples to get
        saved_batch_size = self.batch_size
        self.batch_size = self.n_eval_samples
        paths = self.eval_sampler.obtain_samples(
            itr=epoch,
        )
        self.batch_size = saved_batch_size
        return paths

    @overrides
    def train(self):
        with self.sess.as_default():
            self._init_tensorflow_ops()
            self._train()

    def _train(self):
        n_steps_total = 0
        tf.summary.FileWriter(logger.get_snapshot_dir(), self.sess.graph)
        with self.sess.as_default():
            self._init_training()
            self._start_worker()
            self._switch_to_training_mode()

            observation = self.training_env.reset()
            self.exploration_strategy.reset()
            itr = 0
            path_length = 0
            path_return = 0
            for epoch in range(self.n_epochs):
                logger.push_prefix('Epoch #%d | ' % epoch)
                logger.log("Training started")
                start_time = time.time()
                for n_steps_current_epoch in range(self.epoch_length):
                    with self._eval_then_training_mode():
                        action = self.exploration_strategy.get_action(itr,
                                                                      observation,
                                                                      self.policy)
                    if self.render:
                        self.training_env.render()

                    next_ob, raw_reward, terminal, debug_info = (
                        self.training_env.step(
                            self.process_action(action)
                        )
                    )
                    n_steps_total += 1
                    # Some envs return a Nx1 vector for the observation
                    # TODO(vpong): find a cleaner solution
                    # next_ob = next_ob.squeeze()
                    reward = raw_reward * self.scale_reward
                    path_length += 1
                    path_return += reward

                    self.pool.add_sample(
                        observation,
                        action,
                        reward,
                        terminal,
                        debug_info=debug_info,
                    )
                    if terminal or path_length >= self.max_path_length:
                        self.pool.terminate_episode(next_ob,
                                                    debug_info=debug_info)
                        observation = self.training_env.reset()
                        self.exploration_strategy.reset()
                        self.es_path_returns.append(path_return)
                        path_length = 0
                        path_return = 0
                    else:
                        observation = next_ob

                    if self._can_train():
                        for _ in range(self.n_updates_per_time_step):
                            self._do_training(
                                epoch=epoch,
                                n_steps_total=n_steps_total,
                                n_steps_current_epoch=n_steps_current_epoch,
                            )
                    itr += 1

                logger.log("Training finished. Time: {0}".format(time.time() -
                                                                 start_time))
                if self._can_eval():
                    with self._eval_then_training_mode():
                        start_time = time.time()
                        self.evaluate(epoch, self.es_path_returns)
                        self.es_path_returns = []
                        logger.log(
                            "Eval time: {0}".format(time.time() - start_time))
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()

            self._switch_to_eval_mode()
            self.training_env.terminate()
            self._shutdown_worker()

    def _can_train(self):
        return self.pool.num_can_sample >= self.min_pool_size

    def _can_eval(self):
        return (self.pool.num_can_sample >= self.min_pool_size and
                self.n_eval_samples > 0)

    def _switch_to_training_mode(self):
        """
        Make any updates needed so that the internal networks are in training
        mode.
        :return:
        """
        for network in self._networks:
            network.switch_to_training_mode()

    def _switch_to_eval_mode(self):
        """
        Make any updates needed so that the internal networks are in eval mode.
        :return:
        """
        for network in self._networks:
            network.switch_to_eval_mode()

    @contextmanager
    def _eval_then_training_mode(self):
        """
        Helper method to quickly switch to eval mode and then to training mode

        ```
        # doesn't matter what mode you were in
        with self.eval_then_training_mode():
            # in eval mode
        # in training mode
        :return:
        """
        self._switch_to_eval_mode()
        yield
        self._switch_to_training_mode()

    def _do_training(
            self,
            epoch=None,
            n_steps_total=None,
            n_steps_current_epoch=None,
    ):
        ops = self._get_training_ops(
            epoch=epoch,
            n_steps_total=n_steps_total,
            n_steps_current_epoch=n_steps_current_epoch,
        )
        minibatch = self._sample_minibatch()
        feed_dict = self._update_feed_dict_from_batch(minibatch)
        if isinstance(ops[0], list):
            for op in ops:
                self.sess.run(op, feed_dict=feed_dict)
        else:
            self.sess.run(ops, feed_dict=feed_dict)

    def _sample_minibatch(self):
        return self.pool.random_batch(self.batch_size, flatten=True)

    def _update_feed_dict_from_batch(self, batch):
        return self._update_feed_dict(
            rewards=batch['rewards'],
            terminals=batch['terminals'],
            obs=batch['observations'],
            actions=batch['actions'],
            next_obs=batch['next_observations'],
        )

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.training_env,
            epoch=epoch,
            policy=self.policy,
            es=self.exploration_strategy,
        )

    def evaluate(self, epoch, es_path_returns):
        """
        Perform evaluation for this algorithm.

        It's recommended
        :param epoch: The epoch number.
        :param es_path_returns: List of path returns from explorations strategy
        :return: Dictionary of statistics.
        """
        logger.log("Collecting samples for evaluation")
        paths = self._sample_paths(epoch)
        self.log_diagnostics(paths)

        returns = [sum(path["rewards"]) for path in paths]
        average_returns = np.mean(returns)
        statistics = OrderedDict([
            ('Epoch', epoch),
            ('AverageReturn', average_returns),
        ])
        self._last_average_returns = average_returns

        discounted_returns = [
            special.discount_return(path["rewards"], self.discount)
            for path in paths
        ]
        rewards = np.hstack([path["rewards"] for path in paths])
        statistics.update(create_stats_ordered_dict('Rewards', rewards))
        statistics.update(create_stats_ordered_dict('Returns', returns))
        statistics.update(create_stats_ordered_dict('DiscountedReturns',
                                                    discounted_returns))
        if len(es_path_returns) > 0:
            statistics.update(create_stats_ordered_dict('TrainingReturns',
                                                        es_path_returns))

        statistics.update(self._statistics_from_paths(paths))

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)

    @property
    def final_score(self) -> float:
        """
        :return: The score after training. The objective is to MAXIMIZE this
        value.
        """
        return self._last_average_returns

    @property
    @abc.abstractmethod
    def _networks(self) -> Iterable[NeuralNetwork]:
        """
        :return: List of networks used in the algorithm.

        It's crucial that this list is up to date!
        """
        pass

    @abc.abstractmethod
    def _init_tensorflow_ops(self):
        """
        Method to be called in the initialization of the class. After this
        method is called, the train() method should work.
        :return: None
        """
        return

    @abc.abstractmethod
    def _init_training(self):
        """
        Method to be called at the start of training.
        :return: None
        """
        return

    @abc.abstractmethod
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
        return

    @abc.abstractmethod
    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs,
                          **kwargs):
        """
        :param rewards: np.ndarray, shape BATCH_SIZE
        :param terminals: np.ndarray, shape BATCH_SIZE
        :param obs: np.ndarray, shape BATCH_SIZE x flat observation dim
        :param actions: np.ndarray, shape BATCH_SIZE x flat action dim
        :param next_obs: np.ndarray, shape BATCH_SIZE x flat observation dim
        :return: feed_dict needed for the ops returned by get_training_ops.
        """
        return

    @abc.abstractmethod
    def _statistics_from_paths(self, paths) -> OrderedDict:
        """

        :param paths: List of paths, where a path (AKA trajectory) is the
        output or rllab.sampler.utils.rollout.
        :return: OrderedDict, where
            key = statistics label (string)
            value = statistics value
        """
        pass

    def process_action(self, raw_action):
        """
        Process the action outputted by the policy before giving it to the
        environment.

        :param raw_action:
        :return:
        """
        return raw_action
