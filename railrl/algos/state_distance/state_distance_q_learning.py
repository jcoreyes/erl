import time

import pickle
from collections import OrderedDict

import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.torch.ddpg import DDPG, np_to_pytorch_batch
from rllab.misc import logger, tensor_utils


class MultigoalSimplePathSampler(object):
    def __init__(self, env, policy, max_samples, max_path_length, discount):
        self.env = env
        self.policy = policy
        self.max_samples = max_samples
        self.max_path_length = max_path_length
        self.discount = discount

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self):
        paths = []
        goal = self.env.sample_goal_states(1)[0]
        for _ in range(self.max_samples // self.max_path_length):
            paths.append(multitask_rollout(
                self.env,
                self.policy,
                goal,
                self.discount,
                max_path_length=self.max_path_length,
            ))
        return paths


class StateDistanceQLearning(DDPG):
    def __init__(
            self,
            env,
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
            **kwargs
    ):
        env = pickle.loads(pickle.dumps(env))
        eval_sampler = MultigoalSimplePathSampler(
            env=env,
            policy=policy,
            max_samples=num_steps_per_eval,
            max_path_length=max_path_length,
            discount=discount,
        )
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

        self.use_new_data = use_new_data
        if not self.use_new_data:
            self.replay_buffer = replay_buffer
        self.goal_state = None

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
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                logger.log("Done evaluating")
                logger.pop_prefix()

    def reset_env(self):
        self.exploration_strategy.reset()
        self.exploration_policy.reset()
        self.goal_state = self.training_env.sample_goal_states(1)[0]
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
        new_rewards = self.env.compute_rewards(
            batch['observations'],
            batch['actions'],
            batch['next_observations'],
            goal_states,
        )
        batch['goal_states'] = goal_states
        batch['rewards'] = new_rewards
        torch_batch = np_to_pytorch_batch(batch)
        return torch_batch

    def sample_goal_states(self, batch_size):
        if self.sample_goals_from == 'environment':
            return self.env.sample_goal_states(batch_size)
        elif self.sample_goals_from == 'replay_buffer':
            replay_buffer = self.replay_buffer.get_replay_buffer(training=True)
            batch = replay_buffer.random_batch(batch_size)
            obs = batch['observations']
            return self.env.convert_obs_to_goal_states(obs)

    def _paths_to_np_batch(self, paths):
        batch = super()._paths_to_np_batch(paths)
        batch_size = len(batch['observations'])
        goal_states = self.sample_goal_states(batch_size)
        new_rewards = self.env.compute_rewards(
            batch['observations'],
            batch['actions'],
            batch['next_observations'],
            goal_states,
        )
        batch['goal_states'] = goal_states
        batch['rewards'] = new_rewards
        return batch

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

        statistics['QF Loss Mean Validation - Train Gap'] = (
            statistics['Validation QF Loss Mean']
            - statistics['Train QF Loss Mean']
        )
        statistics['Bellman Errors Max Validation - Train Gap'] = (
            statistics['Validation Bellman Errors Max']
            - statistics['Train Bellman Errors Max']
        )
        statistics['Discount Factor'] = self.discount
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        paths = self._sample_paths(epoch)
        self.log_diagnostics(paths)

        logger.dump_tabular(with_prefix=False, with_timestamp=False)

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
            replay_buffer=self.replay_buffer,
        )


def rollout_with_goal(env, agent, goal, max_path_length=np.inf, animated=False):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    # o = np.hstack((o, goal))
    # o = (o, goal)
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
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
        # o = np.hstack((o, goal))
        # o = (o, goal)
        if animated:
            env.render()

    return dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
        agent_infos=np.array(agent_infos),
        env_infos=np.array(env_infos),
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
