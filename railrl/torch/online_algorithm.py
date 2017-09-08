import abc
import pickle
import time

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.envs.wrappers import convert_gym_space
from railrl.misc.ml_util import ConstantSchedule
from railrl.misc.rllab_util import get_table_key_set, \
    save_extra_data_to_snapshot_dir
from railrl.policies.base import SerializablePolicy
from rllab.algos.base import RLAlgorithm
from rllab.misc import logger, tensor_utils
from rllab.sampler import parallel_sampler
from rllab.sampler.utils import rollout


class SimplePathSampler(object):
    """
    Sample things in another thread by serializing the policy and environment.
    Only one thread is used.
    """
    def __init__(self, env, policy, max_samples, max_path_length):
        self.env = env
        self.policy = policy
        self.max_samples = max_samples
        self.max_path_length = max_path_length

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)

    def shutdown_worker(self):
        parallel_sampler.terminate_task()

    def obtain_samples(self):
        cur_params = self.policy.get_param_values()
        return parallel_sampler.sample_paths(
            policy_params=cur_params,
            max_samples=self.max_samples,
            max_path_length=self.max_path_length,
        )


class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_samples, max_path_length):
        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length
        self.num_rollouts = max_samples // self.max_path_length
        assert self.num_rollouts > 0, "Need max_samples >= max_path_length"

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self):
        return [
            rollout(self.env, self.policy, max_path_length=self.max_path_length)
            for _ in range(self.num_rollouts)
        ]


class OnlineAlgorithm(RLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            exploration_policy: SerializablePolicy,
            exploration_strategy,
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            scale_reward=1,
            render=False,
            save_exploration_path_period=1,
            sample_with_training_env=False,
            epoch_discount_schedule=None,
            eval_sampler=None,
    ):
        self.training_env = env
        self.exploration_policy = exploration_policy
        self.exploration_strategy = exploration_strategy
        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.scale_reward = scale_reward
        self.render = render
        self.save_exploration_path_period = save_exploration_path_period
        self.sample_with_training_env = sample_with_training_env
        if epoch_discount_schedule is None:
            epoch_discount_schedule = ConstantSchedule(self.discount)
        self.epoch_discount_schedule = epoch_discount_schedule

        self.action_space = convert_gym_space(env.action_space)
        self.obs_space = convert_gym_space(env.observation_space)

        if eval_sampler is None:
            # TODO: Remove flag and force to set eval_sampler
            if self.sample_with_training_env:
                self.env = pickle.loads(pickle.dumps(self.training_env))
                self.eval_sampler = SimplePathSampler(
                    env=env,
                    policy=exploration_policy,
                    max_samples=num_steps_per_eval,
                    max_path_length=max_path_length,
                )
            else:
                self.env = env
                self.eval_sampler = InPlacePathSampler(
                    env=env,
                    policy=exploration_policy,
                    max_samples=num_steps_per_eval,
                    max_path_length=max_path_length,
                )
        else:
            self.eval_sampler = eval_sampler
            self.env = eval_sampler.env
        self.replay_buffer = EnvReplayBuffer(
            self.replay_buffer_size,
            self.env,
        )

        self.final_score = 0

    @abc.abstractmethod
    def cuda(self):
        pass

    def reset_env(self):
        self.exploration_strategy.reset()
        self.exploration_policy.reset()
        return self.training_env.reset()

    def get_action_and_info(self, n_steps_total, observation):
        return self.exploration_strategy.get_action(
            n_steps_total,
            observation,
            self.exploration_policy,
        )

    def train(self, start_epoch=0):
        n_steps_total = 0
        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        observation = self.reset_env()
        path_length = 0
        num_paths_total = 0
        self._start_worker()
        self.training_mode(False)
        old_table_keys = None
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        for epoch in range(start_epoch, self.num_epochs):
            self.discount = self.epoch_discount_schedule.get_value(epoch)
            logger.push_prefix('Iteration #%d | ' % epoch)
            start_time = time.time()
            exploration_paths = []
            for _ in range(self.num_steps_per_epoch):
                action, agent_info = self.get_action_and_info(
                    n_steps_total,
                    observation,
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
                    observations.append(self.obs_space.flatten(observation))
                    rewards.append(reward)
                    terminals.append(terminal)
                    actions.append(self.action_space.flatten(action))
                    agent_infos.append(agent_info)
                    env_infos.append(env_info)

                self.replay_buffer.add_sample(
                    observation,
                    action,
                    reward,
                    terminal,
                    agent_info=agent_info,
                    env_info=env_info,
                )
                if terminal or path_length >= self.max_path_length:
                    self.replay_buffer.terminate_episode(
                        next_ob,
                        terminal,
                        agent_info=agent_info,
                        env_info=env_info,
                    )
                    observation = self.reset_env()
                    path_length = 0
                    num_paths_total += 1
                    self.handle_rollout_ending(n_steps_total)
                    if len(observations) > 0:
                        exploration_paths.append(dict(
                            observations=tensor_utils.stack_tensor_list(
                                observations),
                            actions=tensor_utils.stack_tensor_list(actions),
                            rewards=tensor_utils.stack_tensor_list(rewards),
                            terminals=tensor_utils.stack_tensor_list(terminals),
                            agent_infos=tensor_utils.stack_tensor_dict_list(
                                agent_infos),
                            env_infos=tensor_utils.stack_tensor_dict_list(
                                env_infos),
                        ))
                        observations = []
                        actions = []
                        rewards = []
                        terminals = []
                        agent_infos = []
                        env_infos = []
                else:
                    observation = next_ob

                if self._can_train():
                    self.training_mode(True)
                    self._do_training(n_steps_total=n_steps_total)
                    self.training_mode(False)

            train_time = time.time() - start_time
            if self._can_evaluate(exploration_paths):
                start_time = time.time()
                self.evaluate(epoch, exploration_paths)
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                save_extra_data_to_snapshot_dir(
                    self.get_extra_data_to_save(epoch),
                )
                table_keys = get_table_key_set(logger)
                if old_table_keys is not None:
                    assert table_keys == old_table_keys, (
                        "Table keys cannot change from iteration to iteration."
                    )
                old_table_keys = table_keys
                logger.dump_tabular(with_prefix=False, with_timestamp=False)
                logger.log("Eval Time: {0}".format(time.time() - start_time))
            else:
                logger.log("Skipping eval for now.")
            if self._can_train():
                logger.log("Training Time: {0}".format(train_time))
            else:
                logger.log("Not training yet. Time: {}".format(train_time))
            logger.pop_prefix()

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        return dict(
            epoch=epoch,
        )

    def _start_worker(self):
        self.eval_sampler.start_worker()

    def _shutdown_worker(self):
        self.eval_sampler.shutdown_worker()

    def _sample_eval_paths(self, epoch):
        """
        Returns flattened paths.

        :param epoch: Epoch number
        :return: List of dictionaries with these keys:
            observations: np.ndarray, shape BATCH_SIZE x flat observation dim
            actions: np.ndarray, shape BATCH_SIZE x flat action dim
            rewards: np.ndarray, shape BATCH_SIZE
            terminals: np.ndarray, shape BATCH_SIZE
            agent_infos: unsure
            env_infos: unsure
        """
        return self.eval_sampler.obtain_samples()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)

    @abc.abstractmethod
    def evaluate(self, epoch, es_path_returns):
        pass

    def _can_train(self):
        return self.replay_buffer.num_steps_can_sample() >= self.batch_size

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
            env=self.training_env,
        )

    @abc.abstractmethod
    def _do_training(self, n_steps_total):
        pass

    def training_mode(self, mode):
        self.exploration_policy.train(mode)

    def handle_rollout_ending(self, n_steps_total):
        pass

    def _can_evaluate(self, exploration_paths):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :param exploration_paths: List of paths taken while exploring.
        :return:
        """
        return True
