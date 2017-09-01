import pickle
from collections import OrderedDict, Iterable

import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.rllab_util import get_average_returns
from railrl.torch.ddpg import DDPG, np_to_pytorch_batch
from railrl.misc.tensorboard_logger import TensorboardLogger
from rllab.misc import logger


class StateDistanceQLearning(DDPG):
    def __init__(
            self,
            env: MultitaskEnv,
            qf,
            policy,
            replay_buffer=None,
            num_epochs=100,
            num_steps_per_epoch=100,
            sample_goals_from='environment',
            sample_discount=False,
            num_steps_per_eval=1000,
            max_path_length=1000,
            discount=0.99,
            exploration_strategy=None,
            use_new_data=False,
            num_updates_per_env_step=1,
            num_steps_per_tensorboard_update=None,
            prob_goal_state_is_next_state=0,
            termination_threshold=0,
            save_replay_buffer=False,
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
            sample_discount=sample_discount,
        )
        self.num_goals_for_eval = num_steps_per_eval // max_path_length + 1
        super().__init__(
            env,
            qf,
            policy,
            exploration_strategy=exploration_strategy,
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

        self.use_new_data = use_new_data
        if not self.use_new_data:
            self.replay_buffer = replay_buffer
        self.goal_state = None
        if self.num_steps_per_tensorboard_update is not None:
            self.tb_logger = TensorboardLogger(logger.get_snapshot_dir())

    def train(self):
        if self.use_new_data:
            return super().train()
        else:
            num_batches_total = 0
            for epoch in range(self.num_epochs):
                self.discount = self.epoch_discount_schedule.get_value(epoch)
                for _ in range(self.num_steps_per_epoch):
                    self.training_mode(True)
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
        self.exploration_strategy.reset()
        self.exploration_policy.reset()
        self.goal_state = self.sample_goal_state_for_rollout()
        self.training_env.set_goal(self.goal_state)
        return self.training_env.reset()

    def get_action_and_info(self, n_steps_total, observation):
        return self.exploration_strategy.get_action(
            n_steps_total,
            (observation, self.goal_state, self.discount),
            self.exploration_policy,
        )

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
        goal_state = self.sample_goal_states(1)[0]
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
        statistics = OrderedDict()
        train_batch = self.get_batch(training=True)
        statistics.update(self._statistics_from_batch(train_batch, "Train"))
        validation_batch = self.get_batch(training=False)
        statistics.update(
            self._statistics_from_batch(validation_batch, "Validation")
        )

        statistics['Discount Factor'] = self.discount

        paths = self._sample_paths(epoch)
        statistics.update(self._statistics_from_paths(paths, "Test"))
        average_returns = get_average_returns(paths)
        statistics['AverageReturn'] = average_returns

        statistics['QF Loss Mean Validation - Train Gap'] = (
            statistics['Validation QF Loss Mean']
            - statistics['Train QF Loss Mean']
        )
        statistics['QF Loss Mean Test - Validation Gap'] = (
            statistics['Test QF Loss Mean']
            - statistics['Validation QF Loss Mean']
        )
        statistics['Policy Loss Mean Validation - Train Gap'] = (
            statistics['Validation Policy Loss Mean']
            - statistics['Train Policy Loss Mean']
        )
        statistics['Policy Loss Mean Test - Validation Gap'] = (
            statistics['Test Policy Loss Mean']
            - statistics['Validation Policy Loss Mean']
        )

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        self.log_diagnostics(paths)

    def _sample_paths(self, epoch):
        self.eval_sampler.set_discount(self.discount)
        return super()._sample_paths(epoch)

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
        bellman_errors = (y_pred - y_target)**2
        qf_loss = self.qf_criterion(y_pred, y_target)

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('QF Outputs', q_output),
            ('Bellman Errors', bellman_errors),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
            ('QF Loss', qf_loss),
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
        )
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        return data_to_save


class HorizonFedStateDistanceQLearning(StateDistanceQLearning):
    """
    Hacky solution: just use discount in place of max_num_steps_left.
    """
    def get_train_dict(self, batch):
        rewards = batch['rewards']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goal_states = batch['goal_states']

        batch_size = obs.size()[0]
        num_steps_left_np = np.random.randint(
            1, self.discount+1, (batch_size, 1)
        )
        num_steps_left = ptu.np_to_var(num_steps_left_np)
        terminals_np = (num_steps_left_np == 1).astype(int)
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
        next_actions = self.target_policy(next_obs, goal_states, num_steps_left)
        target_q_values = self.target_qf(
            next_obs,
            next_actions,
            goal_states,
            num_steps_left - 1,  # Important! Else QF will (probably) blow up
        )
        y_target = rewards + (1. - terminals) * target_q_values

        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_pred = self.qf(obs, actions, goal_states, num_steps_left)
        bellman_errors = (y_pred - y_target)**2
        qf_loss = self.qf_criterion(y_pred, y_target)

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('QF Outputs', q_output),
            ('Bellman Errors', bellman_errors),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
            ('QF Loss', qf_loss),
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
        env, agent, goal, discount,
        max_path_length=np.inf,
        animated=False,
        combine_goal_and_obs=False,
):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    env.set_goal(goal)
    o = env.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        if combine_goal_and_obs:
            a, agent_info = agent.get_action(np.hstack((o, goal)))
        else:
            a, agent_info = agent.get_action(o, goal, discount)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()

    goal_expanded = np.expand_dims(goal, axis=0)
    # goal_expanded.shape == 1 x goal_dim
    goal_states = goal_expanded.repeat(len(observations), 0)
    # goal_states.shape == path_length x goal_dim
    return dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
        agent_infos=np.array(agent_infos),
        env_infos=np.array(env_infos),
        goal_states=goal_states,
    )
