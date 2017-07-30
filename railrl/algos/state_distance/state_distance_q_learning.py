from collections import OrderedDict

import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.torch.ddpg import DDPG, np_to_pytorch_batch
from rllab.misc import logger


class StateDistanceQLearning(DDPG):
    def __init__(
            self,
            *args,
            replay_buffer=None,
            num_epochs=100,
            num_batches_per_epoch=100,
            sample_goals_from='environment',
            sample_discount=False,
            **kwargs
    ):
        super().__init__(*args, exploration_strategy=None, **kwargs)
        self.num_epochs = num_epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        assert sample_goals_from in ['environment', 'replay_buffer']
        self.sample_goals_from = sample_goals_from
        self.replay_buffer = replay_buffer
        self.sample_discount = sample_discount

    def train(self):
        num_batches_total = 0
        for epoch in range(self.num_epochs):
            self.discount = self.epoch_discount_schedule.get_value(epoch)
            for _ in range(self.num_batches_per_epoch):
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
        batch['observations'] = np.hstack((batch['observations'], goal_states))
        batch['next_observations'] = np.hstack((
            batch['next_observations'], goal_states
        ))
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
            return self.env.convert_obs_to_goal_state(obs)

    def reset_env(self):
        self.exploration_strategy.reset()
        self.exploration_policy.reset()
        self.policy.reset()
        return self.training_env.reset()

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
        batch['observations'] = np.hstack((batch['observations'], goal_states))
        batch['next_observations'] = np.hstack((
            batch['next_observations'], goal_states
        ))
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

        statistics['QF Loss Validation - Train Gap'] = (
            statistics['Validation QF Loss Mean']
            - statistics['Train QF Loss Mean']
        )
        statistics['Discount Factor'] = self.discount
        for key, value in statistics.items():
            logger.record_tabular(key, value)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    def get_train_dict(self, batch):
        if not self.sample_discount:
            return super().get_train_dict(batch)

        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        batch_size = obs.size()[0]
        discount_np = np.random.uniform(0, 1, (batch_size, 1))
        discount = ptu.Variable(ptu.from_numpy(discount_np).float())
        """
        Policy operations.
        """
        policy_actions = self.policy(obs)
        q_output = self.qf(obs, policy_actions, discount)
        policy_loss = - q_output.mean()

        """
        Critic operations.
        """
        next_actions = self.target_policy(next_obs)
        target_q_values = self.target_qf(
            next_obs,
            next_actions,
            discount,
        )
        y_target = rewards + (1. - terminals) * discount * target_q_values

        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_pred = self.qf(obs, actions, discount)
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


def rollout_with_goal(env, agent, goal, max_path_length=np.inf, animated=False):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    o = np.hstack((o, goal))
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
        o = np.hstack((o, goal))
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