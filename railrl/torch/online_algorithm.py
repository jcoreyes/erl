import abc
import time

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.exploration_strategies.noop import NoopStrategy
from rllab.algos.base import RLAlgorithm
from rllab.algos.batch_polopt import BatchSampler
from rllab.misc import logger


class OnlineAlgorithm(RLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            exploration_strategy=None,
            subtraj_length=None,
            num_epochs=100,
            num_steps_epoch=10000,
            batch_size=1024,
    ):
        self.training_env = env
        self.env = env
        self.action_dim = int(env.action_space.flat_dim)
        self.obs_dim = int(env.observation_space.flat_dim)
        self.subtraj_length = subtraj_length

        self.exploration_strategy = exploration_strategy or NoopStrategy()
        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_epoch
        self.batch_size = batch_size
        self.max_path_length = 1000
        self.n_eval_samples = 1000
        self.render = False
        self.scale_reward = 1
        self.pool = EnvReplayBuffer(
            10000,
            self.env,
        )
        self.discount = 1.

        self.scope = None  # Necessary for BatchSampler
        self.whole_paths = True  # Also for BatchSampler
        # noinspection PyTypeChecker
        self.eval_sampler = BatchSampler(self)

        self.policy = None

    def train(self):
        n_steps_total = 0
        observation = self.training_env.reset()
        self.exploration_strategy.reset()
        path_return = 0
        es_path_returns = []
        self._start_worker()
        for epoch in range(self.num_epochs):
            logger.push_prefix('Iteration #%d | ' % epoch)
            start_time = time.time()
            for _ in range(self.num_steps_per_epoch):
                action, agent_info = (
                    self.exploration_strategy.get_action(
                        n_steps_total,
                        observation,
                        self.policy,
                    )
                )
                if self.render:
                    self.training_env.render()

                next_ob, raw_reward, terminal, env_info = (
                    self.training_env.step(action)
                )
                n_steps_total += 1
                reward = raw_reward * self.scale_reward
                path_return += reward

                self.pool.add_sample(
                    observation,
                    action,
                    reward,
                    terminal,
                    agent_info=agent_info,
                    env_info=env_info,
                )
                if terminal:
                    self.pool.terminate_episode(
                        next_ob,
                        agent_info=agent_info,
                        env_info=env_info,
                    )
                    observation = self.training_env.reset()
                    self.exploration_strategy.reset()
                    es_path_returns.append(path_return)
                    path_return = 0
                else:
                    observation = next_ob

                if self._can_train(n_steps_total):
                    self._do_training(n_steps_total=n_steps_total)

            logger.log(
                "Training Time: {0}".format(time.time() - start_time)
            )
            start_time = time.time()
            self.evaluate(epoch, es_path_returns)
            es_path_returns = []
            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
            logger.log("Eval Time: {0}".format(time.time() - start_time))
            logger.pop_prefix()

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

    def _get_other_statistics(self):
        return {}

    def _statistics_from_paths(self, paths):
        return {}

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)

    @abc.abstractmethod
    def evaluate(self, epoch, es_path_returns):
        pass

    def _can_train(self, n_steps_total):
        return self.pool.num_can_sample() >= self.batch_size

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            policy=self.policy,
            env=self.training_env,
        )

    @abc.abstractmethod
    def _do_training(self, n_steps_total):
        pass


