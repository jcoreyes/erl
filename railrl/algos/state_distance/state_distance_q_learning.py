import math
import pickle
import time
import torch
from collections import OrderedDict

import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.ml_util import StatConditionalSchedule
from railrl.misc import rllab_util
from railrl.policies.state_distance import UniversalPolicy
from railrl.samplers.util import rollout
from railrl.torch.ddpg import DDPG
from railrl.torch.algos.util import np_to_pytorch_batch
from railrl.torch.algos.eval import get_difference_statistics
from railrl.misc.tensorboard_logger import TensorboardLogger
from railrl.torch.state_distance.exploration import UniversalExplorationPolicy
from rllab.misc import logger


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
            sample_goals_from='environment',
            sample_discount=False,
            num_steps_per_eval=1000,
            max_path_length=1000,
            discount=0.99,
            use_new_data=False,
            num_updates_per_env_step=1,
            num_steps_per_tensorboard_update=None,
            prob_goal_state_is_next_state=0,
            termination_threshold=0,
            save_replay_buffer=False,
            save_algorithm=False,
            **kwargs
    ):
        env = pickle.loads(pickle.dumps(env))
        eval_sampler = MultigoalSimplePathSampler(
            env=env,
            policy=policy,
            max_samples=num_steps_per_eval,
            max_path_length=max_path_length,
            discount=discount,
            goal_sampling_function=self.sample_goal_state_for_rollout,
            # TODO(vitchyr): create a sample_discount_for_rollout function
            sample_discount=sample_discount,
        )
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
            **kwargs,
        )
        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        assert sample_goals_from in ['environment', 'replay_buffer']
        self.sample_goals_from = sample_goals_from
        self.sample_discount = sample_discount
        self.num_updates_per_env_step = num_updates_per_env_step
        self.num_steps_per_tensorboard_update = num_steps_per_tensorboard_update
        self.prob_goal_state_is_next_state = prob_goal_state_is_next_state
        self.termination_threshold = termination_threshold
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm

        self.use_new_data = use_new_data
        if not self.use_new_data:
            self.replay_buffer = replay_buffer
        self.goal_state = None
        if self.num_steps_per_tensorboard_update is not None:
            self.tb_logger = TensorboardLogger(logger.get_snapshot_dir())
        self.start_time = time.time()

    def train(self, **kwargs):
        self.start_time = time.time()
        if self.use_new_data:
            return super().train()
        else:
            num_batches_total = 0
            for epoch in range(self.num_epochs):
                self.discount = self.epoch_discount_schedule.get_value(epoch)
                self.training_mode(True)
                for _ in range(self.num_steps_per_epoch):
                    self._do_training(n_steps_total=num_batches_total)
                    num_batches_total += 1
                logger.push_prefix('Iteration #%d | ' % epoch)
                self.training_mode(False)
                self.evaluate(epoch, None)
                logger.dump_tabular(with_prefix=False, with_timestamp=False)
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                logger.log("Done evaluating")
                logger.pop_prefix()

    def _do_training(self, n_steps_total):
        for _ in range(self.num_updates_per_env_step):
            super()._do_training(n_steps_total)

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

    def reset_env(self):
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
        goal_states = self.sample_goal_states(batch_size)
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
        batch['rewards'] = self.env.compute_rewards(
            batch['observations'],
            batch['actions'],
            batch['next_observations'],
            goal_states,
        )
        torch_batch = np_to_pytorch_batch(batch)
        return torch_batch

    def sample_goal_states(self, batch_size):
        if self.sample_goals_from == 'environment':
            return self.env.sample_goal_states(batch_size)
        elif self.sample_goals_from == 'replay_buffer':
            replay_buffer = self.replay_buffer.get_replay_buffer(training=True)
            if replay_buffer.num_steps_can_sample() == 0:
                # If there's nothing in the replay...just give all zeros
                return np.zeros((batch_size, self.env.goal_dim))
            batch = replay_buffer.random_batch(batch_size)
            obs = batch['observations']
            return self.env.convert_obs_to_goal_states(obs)
        else:
            raise Exception("Invalid `sample_goals_from`: {}".format(
                self.sample_goals_from
            ))

    def sample_goal_state_for_rollout(self):
        # Always sample goal states from the environment to prevent the
        # degenerate solution where the policy just learns to stay at a fixed
        # location.
        goal_state = self.env.sample_goal_states(1)[0]
        goal_state = self.env.modify_goal_state_for_rollout(goal_state)
        return goal_state

    def _paths_to_np_batch(self, paths):
        np_batch = super()._paths_to_np_batch(paths)
        goal_states = [path["goal_states"] for path in paths]
        np_batch['goal_states'] = np.vstack(goal_states)
        return np_batch

    def evaluate(self, epoch, _):
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

    def _sample_eval_paths(self, epoch):
        self.eval_sampler.set_discount(self.discount)
        return super()._sample_eval_paths(epoch)

    def get_train_dict(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goal_states = batch['goal_states']

        batch_size = obs.size()[0]
        if self.sample_discount:
            discount_np = np.random.uniform(0, self.discount, (batch_size, 1))
        else:
            discount_np = self.discount * np.ones((batch_size, 1))
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
                torch.sum(param**2)
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
        rewards = [path["rewards"].reshape(-1, 1) for path in paths]
        terminals = [path["terminals"].reshape(-1, 1) for path in paths]
        actions = [path["actions"] for path in paths]
        obs = [path["observations"] for path in paths]
        goal_states = [path["goal_states"] for path in paths]
        next_obs = []
        for path in paths:
            next_obs_i = np.vstack((
                path["observations"][1:, :],
                path["final_observation"],
            ))
            next_obs.append(next_obs_i)
        np_batch = dict(
            rewards=np.vstack(rewards),
            terminals=np.vstack(terminals),
            observations=np.vstack(obs),
            actions=np.vstack(actions),
            next_observations=np.vstack(next_obs),
            goal_states=np.vstack(goal_states),
        )
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
            sparse_reward=True,
            fraction_of_taus_set_to_zero=0,
            clamp_q_target_values=False,
            **kwargs
    ):
        """
        I'm reusing discount as tau. Don't feel like renaming everything.

        :param sparse_reward:  The correct interpretation of tau (
        sparse_reward = True) is
        "how far you are from the goal state after tau steps."
        The wrong version just uses tau as a timer.
        :param fraction_of_taus_set_to_zero: This proportion of samples
        taus will be set to zero.
        :param kwargs:
        """
        super().__init__(
            env,
            qf,
            policy,
            exploration_policy,
            **kwargs
        )
        assert 1 >= fraction_of_taus_set_to_zero >= 0
        self.sparse_reward = sparse_reward
        self.fraction_of_taus_set_to_zero = fraction_of_taus_set_to_zero
        self.clamp_q_target_values = clamp_q_target_values

    def get_train_dict(self, batch):
        rewards = batch['rewards']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goal_states = batch['goal_states']

        batch_size = obs.size()[0]
        if self.discount == 0:
            num_steps_left_np = np.zeros((batch_size, 1))
        else:
            num_steps_left_np = np.random.randint(
                0, self.discount + 1, (batch_size, 1)
            )
        if self.fraction_of_taus_set_to_zero > 0:
            num_taus_set_to_zero = int(
                batch_size * self.fraction_of_taus_set_to_zero
            )
            num_steps_left_np[:num_taus_set_to_zero] = 0
        num_steps_left = ptu.np_to_var(num_steps_left_np)
        terminals_np = (num_steps_left_np == 0).astype(int)
        terminals = ptu.np_to_var(terminals_np)

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
        if self.sparse_reward:
            y_target = terminals * rewards + (1. - terminals) * target_q_values
        else:
            y_target = rewards + (1. - terminals) * target_q_values

        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_pred = self.qf(obs, actions, goal_states, num_steps_left)
        bellman_errors = (y_pred - y_target) ** 2
        raw_qf_loss = self.qf_criterion(y_pred, y_target)

        if self.qf_weight_decay > 0:
            reg_loss = self.qf_weight_decay * sum(
                torch.sum(param**2)
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
            self, env, policy, max_samples, max_path_length, discount,
            goal_sampling_function,
            sample_discount=False,
    ):
        self.env = env
        self.policy = policy
        self.max_samples = max_samples
        self.max_path_length = max_path_length
        self.discount = discount
        self.goal_sampling_function = goal_sampling_function
        self.sample_discount = sample_discount

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def set_discount(self, discount):
        self.discount = discount

    def obtain_samples(self):
        paths = []
        for i in range(self.max_samples // self.max_path_length):
            if self.sample_discount:
                discount = np.random.uniform(0, self.discount, 1)[0]
            else:
                discount = self.discount
            goal = self.goal_sampling_function()
            path = multitask_rollout(
                self.env,
                self.policy,
                goal,
                discount,
                max_path_length=self.max_path_length,
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
):
    env.set_goal(goal)
    agent.set_goal(goal)
    agent.set_discount(discount)
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