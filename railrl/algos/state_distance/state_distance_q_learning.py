import math
import time
import torch
from collections import OrderedDict

import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.ml_util import StatConditionalSchedule
from railrl.misc import rllab_util
from railrl.misc.rllab_util import split_paths_to_dict
from railrl.networks.state_distance import DuelingStructuredUniversalQfunction
from railrl.policies.state_distance import UniversalPolicy
from railrl.samplers.util import rollout
from railrl.torch.ddpg import DDPG
from railrl.torch.algos.util import np_to_pytorch_batch
from railrl.torch.algos.eval import (
    get_difference_statistics,
    get_generic_path_information,
)
from railrl.misc.tensorboard_logger import TensorboardLogger
from railrl.torch.state_distance.exploration import UniversalExplorationPolicy
from rllab.misc import logger
import time


class StateDistanceQLearning(DDPG):
    def __init__(
            self,
            env: MultitaskEnv,
            qf,
            policy,
            exploration_policy: UniversalExplorationPolicy = None,
            replay_buffer=None,
            num_epochs=100,
            num_steps_per_epoch=100,
            sample_train_goals_from='environment',
            sample_rollout_goals_from='environment',
            sample_discount=False,
            num_steps_per_eval=1000,
            max_path_length=1000,
            discount=0.99,
            num_updates_per_env_step=1,
            num_steps_per_tensorboard_update=None,
            prob_goal_state_is_next_state=0,
            termination_threshold=0,
            save_replay_buffer=False,
            save_algorithm=False,
            eval_sampler=None,
            **kwargs
    ):
        eval_sampler = eval_sampler or MultigoalSimplePathSampler(
            env=env,
            policy=policy,
            max_samples=num_steps_per_eval,
            max_path_length=max_path_length,
            discount_sampling_function=self._sample_discount_for_rollout,
            goal_sampling_function=self.sample_goal_state_for_rollout,
            cycle_taus_for_rollout=False,
        )
        if sample_train_goals_from == 'her':
            assert isinstance(replay_buffer, SplitReplayBuffer)
            assert isinstance(replay_buffer.train_replay_buffer,
                              HerReplayBuffer)
            assert isinstance(replay_buffer.validation_replay_buffer,
                              HerReplayBuffer)
        self.num_goals_for_eval = num_steps_per_eval // max_path_length + 1
        super().__init__(
            env,
            qf,
            policy,
            exploration_policy=exploration_policy,
            eval_sampler=eval_sampler,
            num_steps_per_eval=num_steps_per_eval,
            discount=discount,
            max_path_length=max_path_length,
            replay_buffer=replay_buffer,
            **kwargs,
        )
        if isinstance(self.qf, DuelingStructuredUniversalQfunction):
            self.qf.set_argmax_policy(self.policy)
            self.target_qf.set_argmax_policy(self.target_policy)
        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        assert sample_train_goals_from in ['environment', 'replay_buffer', 'her']
        assert sample_rollout_goals_from in ['environment', 'replay_buffer']
        self.sample_train_goals_from = sample_train_goals_from
        self.sample_rollout_goals_from = sample_rollout_goals_from
        self.sample_discount = sample_discount
        self.num_updates_per_env_step = num_updates_per_env_step
        self.num_steps_per_tensorboard_update = num_steps_per_tensorboard_update
        self.prob_goal_state_is_next_state = prob_goal_state_is_next_state
        self.termination_threshold = termination_threshold
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm

        self.goal_state = None
        if self.num_steps_per_tensorboard_update is not None:
            self.tb_logger = TensorboardLogger(logger.get_snapshot_dir())
        self.start_time = time.time()

    def _do_training(self, n_steps_total):
        for _ in range(self.num_updates_per_env_step):
            # prev = time.time()
            super()._do_training(n_steps_total)
            # print(time.time()-prev)
        if self.num_steps_per_tensorboard_update is None:
            return

        if n_steps_total % self.num_steps_per_tensorboard_update == 0:
            for name, network in [
                ("QF", self.qf),
                ("Policy", self.policy),
            ]:
                for param_tag, value in network.named_parameters():
                    param_tag = param_tag.replace('.', '/')
                    tag = "{}/{}".format(name, param_tag)
                    self.tb_logger.histo_summary(
                        tag,
                        ptu.get_numpy(value),
                        n_steps_total + 1,
                        bins=100,
                    )
                    self.tb_logger.histo_summary(
                        tag + '/grad',
                        ptu.get_numpy(value.grad),
                        n_steps_total + 1,
                        bins=100,
                    )

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        self.goal_state = self.sample_goal_state_for_rollout()
        self.training_env.set_goal(self.goal_state)
        self.exploration_policy.set_goal(self.goal_state)
        self.exploration_policy.set_discount(self.discount)
        return self.training_env.reset()

    def get_batch(self, training=True):
        replay_buffer = self.replay_buffer.get_replay_buffer(training)
        batch_size = min(
            replay_buffer.num_steps_can_sample(),
            self.batch_size
        )
        batch = replay_buffer.random_batch(batch_size)
        if self.sample_train_goals_from != 'her':
            goal_states = self.sample_goal_states_for_training(batch_size)
            if self.prob_goal_state_is_next_state > 0:
                num_next_states_as_goal_states = int(
                    self.prob_goal_state_is_next_state * batch_size
                )
                goal_states[:num_next_states_as_goal_states] = (
                    batch['next_observations'][:num_next_states_as_goal_states]
                )
            batch['goal_states'] = goal_states
        if self.termination_threshold > 0:
            batch['terminals'] = np.linalg.norm(
                self.env.convert_obs_to_goal_states(
                    batch['next_observations']
                ) - goal_states,
                axis=1,
            ) <= self.termination_threshold
        batch['rewards'] = self.compute_rewards(
            batch['observations'],
            batch['actions'],
            batch['next_observations'],
            goal_states,
        )
        torch_batch = np_to_pytorch_batch(batch)
        return torch_batch

    def compute_rewards(self, obs, actions, next_obs, goal_states):
        return self.env.compute_rewards(
            obs,
            actions,
            next_obs,
            goal_states,
        )

    def sample_goal_states_for_training(self, batch_size):
        if self.sample_goals_from == 'environment':
            return self.env.sample_goal_states_for_training(batch_size)
        elif self.sample_goals_from == 'replay_buffer':
            replay_buffer = self.replay_buffer.get_replay_buffer(training=True)
            if replay_buffer.num_steps_can_sample() == 0:
                # If there's nothing in the replay...just give all zeros
                return np.zeros((batch_size, self.env.goal_dim))
            batch = replay_buffer.random_batch(batch_size)
            obs = batch['observations']
            return self.env.convert_obs_to_goal_states(obs)
        elif self.sample_train_goals_from == 'her':
            raise Exception
        else:
            raise Exception("Invalid `sample_goals_from`: {}".format(
                self.sample_goals_from
            ))

    def sample_goal_state_for_rollout(self):
        if self.sample_rollout_goals_from == 'environment':
            goal_state = self.env.sample_goal_states_for_training(1)[0]
        elif self.sample_rollout_goals_from == 'replay_buffer':
            replay_buffer = self.replay_buffer.get_replay_buffer(training=True)
            if replay_buffer.num_steps_can_sample() == 0:
                # If there's nothing in the replay...just give all zeros
                return np.zeros(self.env.goal_dim)
            batch = replay_buffer.random_batch(0)
            obs = batch['observations']
            goal_state = self.env.convert_obs_to_goal_states(obs)[0]
        else:
            raise Exception("Invalid `sample_goals_from`: {}".format(
                self.sample_rollout_goals_from
            ))
        return self.env.modify_goal_state_for_rollout(goal_state)

    def _sample_discount(self, batch_size):
        if self.sample_discount:
            return np.random.uniform(0, self.discount, (batch_size, 1))
        else:
            return self.discount * np.ones((batch_size, 1))

    def _sample_discount_for_rollout(self):
        return self._sample_discount(1)[0, 0]

    def evaluate(self, epoch, exploration_paths):
        """
        Perform evaluation for this algorithm.

        :param epoch: The epoch number.
        :param exploration_paths: List of dicts, each representing a path.
        """
        logger.log("Collecting samples for evaluation")
        statistics = OrderedDict()
        train_batch = self.get_batch(training=True)
        validation_batch = self.get_batch(training=False)
        test_paths = self._sample_eval_paths(epoch)

        statistics.update(self._statistics_from_batch(train_batch, "Train"))
        statistics.update(
            self._statistics_from_batch(validation_batch, "Validation")
        )
        statistics.update(self._statistics_from_paths(test_paths, "Test"))
        statistics.update(
            get_difference_statistics(
                statistics,
                [
                    'QF Loss Mean',
                    'Policy Loss Mean',
                ],
            )
        )
        statistics.update(get_generic_path_information(
            exploration_paths, self.discount, stat_prefix="Exploration",
        ))
        statistics.update(self._statistics_from_paths(exploration_paths,
                                                      "Exploration"))

        statistics['Discount Factor'] = self.discount

        average_returns = rllab_util.get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns
        statistics['Total Wallclock Time (s)'] = time.time() - self.start_time
        statistics['Epoch'] = epoch

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        self.log_diagnostics(test_paths)

        if isinstance(self.epoch_discount_schedule, StatConditionalSchedule):
            table_dict = rllab_util.get_logger_table_dict()
            # rllab converts things to strings for some reason
            value = float(
                table_dict[self.epoch_discount_schedule.statistic_name]
            )
            self.epoch_discount_schedule.update(value)

    def get_train_dict(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goal_states = batch['goal_states']

        batch_size = obs.size()[0]
        discount_np = self._sample_discount(batch_size)
        discount = ptu.Variable(ptu.from_numpy(discount_np).float())

        """
        Policy operations.
        """
        policy_actions = self.policy(obs, goal_states, discount)
        q_output = self.qf(obs, policy_actions, goal_states, discount)
        policy_loss = - q_output.mean()

        """
        Critic operations.
        """
        next_actions = self.target_policy(next_obs, goal_states, discount)
        target_q_values = self.target_qf(
            next_obs,
            next_actions,
            goal_states,
            discount,
        )
        y_target = rewards + (1. - terminals) * discount * target_q_values

        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_pred = self.qf(obs, actions, goal_states, discount)
        bellman_errors = (y_pred - y_target) ** 2
        raw_qf_loss = self.qf_criterion(y_pred, y_target)

        if self.qf_weight_decay > 0:
            reg_loss = self.qf_weight_decay * sum(
                torch.sum(param ** 2)
                for param in self.qf.regularizable_parameters()
            )
            qf_loss = raw_qf_loss + reg_loss
        else:
            qf_loss = raw_qf_loss

        """
        Target Policy operations if needed
        """
        if self.optimize_target_policy:
            target_policy_actions = self.target_policy(
                obs,
                goal_states,
                discount,
            )
            target_q_output = self.target_qf(
                obs,
                target_policy_actions,
                goal_states,
                discount,
            )
            target_policy_loss = - target_q_output.mean()
        else:
            # Always include the target policy loss so that different
            # experiments are easily comparable.
            target_policy_loss = ptu.FloatTensor([0])

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('QF Outputs', q_output),
            ('Bellman Errors', bellman_errors),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
            ('Unregularized QF Loss', raw_qf_loss),
            ('QF Loss', qf_loss),
            ('Target Policy Loss', target_policy_loss),
        ])

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            policy=self.policy,
            env=self.training_env,
            qf=self.qf,
            discount=self.discount,
        )

    def get_extra_data_to_save(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            env=self.training_env,
            algorithm=self,
        )
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    @staticmethod
    def paths_to_batch(paths):
        np_batch = split_paths_to_dict(paths)
        goal_states = [path["goal_states"] for path in paths]
        np_batch['goal_states'] = np.vstack(goal_states)
        return np_to_pytorch_batch(np_batch)


class HorizonFedStateDistanceQLearning(StateDistanceQLearning):
    """
    Hacky solution: just use discount in place of max_num_steps_left.
    """

    def __init__(
            self,
            env: MultitaskEnv,
            qf,
            policy,
            exploration_policy: UniversalExplorationPolicy = None,
            eval_policy: UniversalPolicy = None,
            num_steps_per_eval=1000,
            discount=1,
            max_path_length=1000,
            sparse_reward=True,
            fraction_of_taus_set_to_zero=0,
            clamp_q_target_values=False,
            cycle_taus_for_rollout=True,
            **kwargs
    ):
        """
        I'm reusing discount as tau. Don't feel like renaming everything.

        :param eval_policy: If None, default to using the DDPG policy for eval
        :param sparse_reward:  The correct interpretation of tau (
        sparse_reward = True) is
        "how far you are from the goal state after tau steps."
        The wrong version just uses tau as a timer.
        :param fraction_of_taus_set_to_zero: This proportion of samples
        taus will be set to zero.
        :param cycle_taus_for_rollout: Decrement tau at each time step when
        collecting data.
        :param kwargs:
        """
        if cycle_taus_for_rollout:
            assert discount > 0
        if eval_policy is None:
            eval_policy = policy
        eval_sampler = MultigoalSimplePathSampler(
            env=env,
            policy=eval_policy,
            max_samples=num_steps_per_eval,
            max_path_length=max_path_length,
            discount_sampling_function=self._sample_discount_for_rollout,
            goal_sampling_function=self.sample_goal_state_for_rollout,
            cycle_taus_for_rollout=cycle_taus_for_rollout,
        )
        super().__init__(
            env,
            qf,
            policy,
            exploration_policy,
            eval_sampler=eval_sampler,
            num_steps_per_eval=num_steps_per_eval,
            discount=discount,
            max_path_length=max_path_length,
            **kwargs
        )
        assert 1 >= fraction_of_taus_set_to_zero >= 0
        self.sparse_reward = sparse_reward
        self.fraction_of_taus_set_to_zero = fraction_of_taus_set_to_zero
        self.clamp_q_target_values = clamp_q_target_values
        self.cycle_taus_for_rollout = cycle_taus_for_rollout
        self._rollout_tau = self.discount

    def _sample_discount(self, batch_size):
        if self.sample_discount:
            return np.random.randint(0, self.discount + 1, (batch_size, 1))
        else:
            return self.discount * np.ones((batch_size, 1))

    def _sample_discount_for_rollout(self):
        if self.cycle_taus_for_rollout:
            return self.discount
        else:
            return self._sample_discount(1)[0, 0]

    def _start_new_rollout(self):
        """
        Implement anything that needs to happen before every rollout.
        :return:
        """
        self._rollout_discount = self.discount
        if not self.cycle_taus_for_rollout:
            return super()._start_new_rollout()

        self.exploration_policy.set_discount(self._rollout_discount)
        return super()._start_new_rollout()

    def _handle_step(
            self,
            num_paths_total,
            observation,
            action,
            reward,
            terminal,
            agent_info,
            env_info,
    ):
        if num_paths_total % self.save_exploration_path_period == 0:
            self._current_path.add_all(
                observations=self.obs_space.flatten(observation),
                rewards=reward,
                terminals=terminal,
                actions=self.action_space.flatten(action),
                agent_infos=agent_info,
                env_infos=env_info,
                goal_states=self.goal_state,
                taus=self._rollout_discount,
            )

        self.replay_buffer.add_sample(
            observation,
            action,
            reward,
            terminal,
            agent_info=agent_info,
            env_info=env_info,
        )

        if self.cycle_taus_for_rollout:
            self._rollout_discount -= 1
            if self._rollout_discount < 0:
                self._rollout_discount = self.discount
            self.exploration_policy.set_discount(self._rollout_discount)

    def _modify_batch_for_training(self, batch):
        obs = batch['observations']
        batch_size = obs.size()[0]
        if self.discount == 0:
            num_steps_left = np.zeros((batch_size, 1))
        else:
            num_steps_left = np.random.randint(
                0, self.discount + 1, (batch_size, 1)
            )
        if self.fraction_of_taus_set_to_zero > 0:
            num_taus_set_to_zero = int(
                batch_size * self.fraction_of_taus_set_to_zero
            )
            num_steps_left[:num_taus_set_to_zero] = 0
        terminals = (num_steps_left == 0).astype(int)
        batch['num_steps_left'] = ptu.np_to_var(num_steps_left)
        batch['terminals'] = ptu.np_to_var(terminals)
        if self.sparse_reward:
            batch['rewards'] = batch['rewards'] * batch['terminals']
        return batch

    def get_train_dict(self, batch):
        batch = self._modify_batch_for_training(batch)
        rewards = batch['rewards']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goal_states = batch['goal_states']
        terminals = batch['terminals']
        num_steps_left = batch['num_steps_left']

        """
        Policy operations.
        """
        policy_actions = self.policy(obs, goal_states, num_steps_left)
        q_output = self.qf(obs, policy_actions, goal_states, num_steps_left)
        policy_loss = - q_output.mean()

        """
        Critic operations.
        """
        next_actions = self.target_policy(
            next_obs,
            goal_states,
            num_steps_left - 1,
        )
        target_q_values = self.target_qf(
            next_obs,
            next_actions,
            goal_states,
            num_steps_left - 1,  # Important! Else QF will (probably) blow up
        )
        if self.clamp_q_target_values:
            target_q_values = torch.clamp(target_q_values, -math.inf, 0)
        y_target = rewards + (1. - terminals) * target_q_values

        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_pred = self.qf(obs, actions, goal_states, num_steps_left)
        bellman_errors = (y_pred - y_target) ** 2
        raw_qf_loss = self.qf_criterion(y_pred, y_target)

        if self.qf_weight_decay > 0:
            reg_loss = self.qf_weight_decay * sum(
                torch.sum(param ** 2)
                for param in self.qf.regularizable_parameters()
            )
            qf_loss = raw_qf_loss + reg_loss
        else:
            qf_loss = raw_qf_loss

        """
        Target Policy operations if needed
        """
        if self.optimize_target_policy:
            target_policy_actions = self.target_policy(
                obs,
                goal_states,
                num_steps_left,
            )
            target_q_output = self.target_qf(
                obs,
                target_policy_actions,
                goal_states,
                num_steps_left,
            )
            target_policy_loss = - target_q_output.mean()
        else:
            # Always include the target policy loss so that different
            # experiments are easily comparable.
            target_policy_loss = ptu.FloatTensor([0])

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('QF Outputs', q_output),
            ('Bellman Errors', bellman_errors),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
            ('Unregularized QF Loss', raw_qf_loss),
            ('QF Loss', qf_loss),
            ('Target Policy Loss', target_policy_loss),
        ])


class MultigoalSimplePathSampler(object):
    def __init__(
            self,
            env,
            policy,
            max_samples,
            max_path_length,
            discount_sampling_function,
            goal_sampling_function,
            cycle_taus_for_rollout=True,
    ):
        self.env = env
        self.policy = policy
        self.max_samples = max_samples
        self.max_path_length = max_path_length
        self.discount_sampling_function = discount_sampling_function
        self.goal_sampling_function = goal_sampling_function
        self.cycle_taus_for_rollout = cycle_taus_for_rollout

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self):
        paths = []
        for i in range(self.max_samples // self.max_path_length):
            discount = self.discount_sampling_function()
            goal = self.goal_sampling_function()
            path = multitask_rollout(
                self.env,
                self.policy,
                goal,
                discount,
                max_path_length=self.max_path_length,
                decrement_discount=self.cycle_taus_for_rollout,
                cycle_tau=self.cycle_taus_for_rollout,
            )
            path_length = len(path['observations'])
            path['goal_states'] = expand_goal(goal, path_length)
            paths.append(path)
        return paths


def expand_goal(goal, path_length):
    return np.repeat(
        np.expand_dims(goal, 0),
        path_length,
        0,
    )


def multitask_rollout(
        env,
        agent: UniversalPolicy,
        goal,
        discount,
        max_path_length=np.inf,
        animated=False,
        decrement_discount=False,
        cycle_tau=False,
):
    env.set_goal(goal)
    agent.set_goal(goal)
    agent.set_discount(discount)
    if decrement_discount:
        assert max_path_length >= discount
        path = rollout_decrement_tau(
            env,
            agent,
            discount,
            max_path_length=max_path_length,
            animated=animated,
            cycle_tau=cycle_tau,
        )
    else:
        path = rollout(
            env,
            agent,
            max_path_length=max_path_length,
            animated=animated,
        )
    goal_expanded = np.expand_dims(goal, axis=0)
    # goal_expanded.shape == 1 x goal_dim
    path['goal_states'] = goal_expanded.repeat(len(path['observations']), 0)
    # goal_states.shape == path_length x goal_dim
    return path


def rollout_decrement_tau(env, agent, init_tau, max_path_length=np.inf,
                          animated=False, cycle_tau=False):
    """
    Decrement tau by one at each time step. If tau < 0, keep it at zero or
    reset it to the init tau.

    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :param cycle_tau: If False, just keep tau equal to zero once it reaches
    zero. Otherwise cycle it.
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    taus = []
    o = env.reset()
    next_o = None
    path_length = 0
    tau = init_tau
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        agent.set_discount(tau)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        taus.append(tau)
        path_length += 1
        tau -= 1
        if tau < 0:
            if cycle_tau:
                tau = init_tau
            else:
                tau = 0
        if d:
            break
        o = next_o
        if animated:
            env.render()
            # input("Press Enter to continue...")

    return dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
        agent_infos=np.array(agent_infos),
        env_infos=np.array(env_infos),
        final_observation=next_o,
        taus=np.array(taus),
    )
