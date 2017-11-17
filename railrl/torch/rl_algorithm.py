import abc
import pickle
import time

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.path import Path
from railrl.envs.wrappers import convert_gym_space
from railrl.misc.rllab_util import (
    get_table_key_set,
    save_extra_data_to_snapshot_dir,
)
from railrl.policies.base import ExplorationPolicy
from railrl.samplers.util import rollout
from rllab.misc import logger
from rllab.sampler import parallel_sampler


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


class RLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            exploration_policy: ExplorationPolicy,
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            scale_reward=1,
            render=False,
            sample_with_training_env=False,
            eval_sampler=None,
            save_replay_buffer=False,
            save_algorithm=False,
            collection_mode='online',
    ):
        assert collection_mode in ['online', 'online-parallel', 'offline']
        self.training_env = pickle.loads(pickle.dumps(env))
        self.exploration_policy = exploration_policy
        self.num_epochs = num_epochs
        self.num_env_steps_per_epoch = num_steps_per_epoch
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.scale_reward = scale_reward
        self.render = render
        self.sample_with_training_env = sample_with_training_env
        self.collection_mode = collection_mode
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm

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

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._epoch_start_time = None
        self._old_table_keys = None
        self._current_path = Path()
        self._exploration_paths = []

    def train(self, start_epoch=0):
        if start_epoch == 0:
            params = self.get_epoch_snapshot(-1)
            logger.save_itr_params(-1, params)
        self.training_mode(False)
        self._n_env_steps_total = start_epoch * self.num_env_steps_per_epoch
        if self.collection_mode == 'online':
            self.train_online(start_epoch=start_epoch)
        elif self.collection_mode == 'online-parallel':
            self.train_parallel(start_epoch=start_epoch)
        elif self.collection_mode == 'offline':
            self.train_offline(start_epoch=start_epoch)
        else:
            raise NotImplementedError("Invalid collection_mode: {}".format(
                self.collection_mode
            ))

    def train_online(self, start_epoch=0):
        self._current_path = Path()
        observation = self._start_new_rollout()
        for epoch in range(start_epoch, self.num_epochs):
            self._start_epoch(epoch)
            for _ in range(self.num_env_steps_per_epoch):
                action, agent_info = self._get_action_and_info(
                    observation,
                )
                if self.render:
                    self.training_env.render()
                next_ob, raw_reward, terminal, env_info = (
                    self.training_env.step(action)
                )
                self._n_env_steps_total += 1
                reward = raw_reward * self.scale_reward
                self._handle_step(
                    observation,
                    action,
                    reward,
                    terminal,
                    agent_info=agent_info,
                    env_info=env_info,
                )
                if terminal or len(self._current_path) >= self.max_path_length:
                    self._handle_rollout_ending(
                        next_ob,
                        terminal,
                        agent_info=agent_info,
                        env_info=env_info,
                    )
                    observation = self._start_new_rollout()
                    if len(self._current_path) > 0:
                        self._exploration_paths.append(
                            self._current_path.get_all_stacked()
                        )
                        self._current_path = Path()
                else:
                    observation = next_ob

                self._try_to_train()

            self._try_to_eval(epoch)
            self._end_epoch()

    def train_parallel(self, start_epoch=0):
        self.training_mode(False)
        n_steps_current_epoch = 0
        epoch = start_epoch
        self._start_epoch(epoch)
        while self._n_env_steps_total <= self.num_epochs * self.num_env_steps_per_epoch:
            path = self.training_env.rollout(
                self.exploration_policy,
                use_exploration_strategy=True,
            )
            if path is not None:
                path['rewards'] *= self.scale_reward
                path_length = len(path['observations'])
                self._n_env_steps_total += path_length
                n_steps_current_epoch += path_length
                self._handle_path(path)

            self._try_to_train()

            # Check if epoch is over
            if n_steps_current_epoch >= self.num_env_steps_per_epoch:
                self._try_to_eval(epoch)
                if self._can_evaluate():
                    logger.record_tabular(
                        "Number of train steps total",
                        self._n_train_steps_total,
                    )
                self._end_epoch()

                epoch += 1
                n_steps_current_epoch = 0
                self._start_epoch(epoch)

    def train_offline(self, start_epoch=0):
        self.training_mode(False)
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        for epoch in range(start_epoch, self.num_epochs):
            self._start_epoch(epoch)
            self._try_to_train()
            self._try_to_offline_eval(epoch)
            self._end_epoch()

    def _try_to_train(self):
        if self._can_train():
            self.training_mode(True)
            self._n_train_steps_total += 1
            self._do_training()
            self.training_mode(False)

    def _try_to_eval(self, epoch):
        save_extra_data_to_snapshot_dir(
            self.get_extra_data_to_save(epoch),
        )
        if self._can_evaluate():
            start_time = time.time()
            self.evaluate(epoch)
            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = get_table_key_set(logger)
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
            logger.log("Eval Time: {0}".format(time.time() - start_time))
        else:
            logger.log("Skipping eval for now.")

    def _try_to_offline_eval(self, epoch):
        start_time = time.time()
        self.offline_evaluate(epoch)
        params = self.get_epoch_snapshot(epoch)
        logger.save_itr_params(epoch, params)
        table_keys = get_table_key_set(logger)
        if self._old_table_keys is not None:
            assert table_keys == self._old_table_keys, (
                "Table keys cannot change from iteration to iteration."
            )
        self._old_table_keys = table_keys
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        logger.log("Eval Time: {0}".format(time.time() - start_time))

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        return (
            len(self._exploration_paths) > 0
            and self.replay_buffer.num_steps_can_sample() >= self.batch_size
        )

    def _can_train(self):
        return self.replay_buffer.num_steps_can_sample() >= self.batch_size

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        return self.exploration_policy.get_action(
            observation,
        )

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        return self.training_env.reset()

    def _handle_path(self, path):
        """
        Naive implementation: just loop through each transition.
        :param path:
        :return:
        """
        for (
            reward,
            terminal,
            action,
            obs,
            agent_info,
            env_info
        ) in zip(
            path["rewards"].reshape(-1, 1),
            path["terminals"].reshape(-1, 1),
            path["actions"],
            path["observations"],
            path["agent_infos"],
            path["env_infos"],
        ):
            self._handle_step(
                obs,
                action,
                reward,
                terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self.replay_buffer.terminate_episode(
            path["final_observation"],
            path["terminals"][-1],
            agent_info=path["agent_infos"][-1],
            env_info=path["env_infos"][-1],
        )
        self._handle_rollout_ending(
            path["final_observation"],
            path["terminals"][-1],
            agent_info=path["agent_infos"][-1],
            env_info=path["env_infos"][-1],
        )

    def _handle_step(
            self,
            observation,
            action,
            reward,
            terminal,
            agent_info,
            env_info,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        self._current_path.add_all(
            observations=self.obs_space.flatten(observation),
            rewards=reward,
            terminals=terminal,
            actions=self.action_space.flatten(action),
            agent_infos=agent_info,
            env_infos=env_info,
        )

        self.replay_buffer.add_sample(
            observation,
            action,
            reward,
            terminal,
            agent_info=agent_info,
            env_info=env_info,
        )

    def _handle_rollout_ending(
            self,
            final_obs,
            terminal,
            agent_info,
            env_info,
    ):
        """
        Implement anything that needs to happen after every rollout.
        """
        self._current_path.add_all(
            final_observation=final_obs,
            increment_path_length=False,
        )
        self.replay_buffer.terminate_episode(
            final_obs,
            terminal,
            agent_info=agent_info,
            env_info=env_info,
        )

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
            env=self.training_env,
        )

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        data_to_save = dict(
            epoch=epoch,
            env=self.training_env,
        )
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def cuda(self):
        """
        Turn cuda on.
        :return:
        """
        pass

    @abc.abstractmethod
    def evaluate(self, epoch):
        """
        Evaluate the policy, e.g. save/print progress.
        :param epoch:
        :return:
        """
        pass

    @abc.abstractmethod
    def offline_evaluate(self, epoch):
        """
        Evaluate without collecting new data.
        :param epoch:
        :return:
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass
