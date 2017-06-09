import abc
import pickle
import time

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.exploration_strategies.noop import NoopStrategy
from rllab.algos.base import RLAlgorithm
from rllab.algos.batch_polopt import BatchSampler
from rllab.misc import logger, tensor_utils


class OnlineAlgorithm(RLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            exploration_strategy=None,
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            pool_size=1000000,
            scale_reward=1,
            use_gpu=False,
            render=False,
            save_exploration_path_period=10,
    ):
        self.training_env = env
        self.exploration_strategy = exploration_strategy
        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.pool_size = pool_size
        self.scale_reward = scale_reward
        self.use_gpu = use_gpu
        self.render = render
        self.save_exploration_path_period = save_exploration_path_period

        self.env = pickle.loads(pickle.dumps(self.training_env))
        self.action_dim = int(env.action_space.flat_dim)
        self.obs_dim = int(env.observation_space.flat_dim)
        self.pool = EnvReplayBuffer(
            self.pool_size,
            self.env,
        )

        self.scope = None  # Necessary for BatchSampler
        self.whole_paths = True  # Also for BatchSampler
        # noinspection PyTypeChecker
        self.eval_sampler = BatchSampler(self)

        self.policy = None  # Subclass must set this.

    def train(self):
        n_steps_total = 0
        observation = self.training_env.reset()
        self.exploration_strategy.reset()
        path_length = 0
        num_paths_total = 0
        self._start_worker()
        self.training_mode(False)
        for epoch in range(self.num_epochs):
            logger.push_prefix('Iteration #%d | ' % epoch)
            start_time = time.time()
            paths = []
            observations = []
            actions = []
            rewards = []
            terminals = []
            agent_infos = []
            env_infos = []
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
                # path_return += reward
                path_length += 1

                if num_paths_total % self.save_exploration_path_period == 0:
                    observations.append(
                        self.training_env.observation_space.flatten(observation))
                    rewards.append(reward)
                    terminals.append(terminal)
                    actions.append(self.training_env.action_space.flatten(action))
                    agent_infos.append(agent_info)
                    env_infos.append(env_info)

                self.pool.add_sample(
                    observation,
                    action,
                    reward,
                    terminal,
                    agent_info=agent_info,
                    env_info=env_info,
                )
                if terminal or path_length >= self.max_path_length:
                    self.pool.terminate_episode(
                        next_ob,
                        terminal,
                        agent_info=agent_info,
                        env_info=env_info,
                    )
                    observation = self.training_env.reset()
                    self.exploration_strategy.reset()
                    path_length = 0
                    num_paths_total += 1
                    self.handle_rollout_ending(n_steps_total)
                    if len(observations) > 0:
                        paths.append(dict(
                            observations=tensor_utils.stack_tensor_list(observations),
                            actions=tensor_utils.stack_tensor_list(actions),
                            rewards=tensor_utils.stack_tensor_list(rewards),
                            terminals=tensor_utils.stack_tensor_list(terminals),
                            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
                            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
                        ))
                else:
                    observation = next_ob

                if self._can_train():
                    self.training_mode(True)
                    self._do_training(n_steps_total=n_steps_total)
                    self.training_mode(False)

            if self._can_train():
                logger.log(
                    "Training Time: {0}".format(time.time() - start_time)
                )
            else:
                logger.log("Not training yet. Time: {}".format(
                    time.time() - start_time)
                )
            start_time = time.time()
            self.evaluate(epoch, paths)
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
        self.batch_size = self.num_steps_per_eval
        paths = self.eval_sampler.obtain_samples(
            itr=epoch,
        )
        self.batch_size = saved_batch_size
        return paths

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)

    @abc.abstractmethod
    def evaluate(self, epoch, es_path_returns):
        pass

    def _can_train(self):
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

    def training_mode(self, mode):
        self.policy.train(mode)

    def handle_rollout_ending(self, n_steps_total):
        pass


