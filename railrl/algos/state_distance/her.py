import torch
from collections import OrderedDict

from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.envs.wrappers import convert_gym_space
from railrl.torch.ddpg import DDPG


class HER(DDPG):
    def __init__(
            self,
            env,
            qf,
            policy,
            exploration_policy,
            replay_buffer,
            epsilon=1e-4,
            **kwargs
    ):
        assert isinstance(replay_buffer, HerReplayBuffer)
        super().__init__(
            env, qf, policy, exploration_policy,
            replay_buffer=replay_buffer,
            **kwargs
        )
        self.epsilon = epsilon
        assert self.qf_weight_decay == 0
        assert not self.optimize_target_policy
        assert self.residual_gradient_weight == 0

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        self.goal_state = self.env.sample_goal_state_for_rollout()
        self.training_env.set_goal(self.goal_state)
        self.exploration_policy.set_goal(self.goal_state)
        return self.training_env.reset()

    def get_batch(self, training=True):
        batch = super().get_batch(training=training)
        batch['rewards'] = (torch.abs(
            self.env.convert_obs_to_goal_state(batch['observations'])
            - batch['goals_states']
        ) < self.epsilon).float()
        return batch

    def get_train_dict(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goal_states = batch['goal_states']

        """
        Policy operations.
        """
        policy_actions = self.policy(obs, goal_states)
        q_output = self.qf(obs, policy_actions, goal_states)
        policy_loss = - q_output.mean()

        """
        Critic operations.
        """
        next_actions = self.target_policy(next_obs, goal_states)
        # speed up computation by not backpropping these gradients
        next_actions.detach()
        target_q_values = self.target_qf(
            next_obs,
            next_actions,
            goal_states,
        )
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        y_target = torch.clamp(y_target, -1./(1-self.discount), 0)
        y_pred = self.qf(obs, actions, goal_states)
        bellman_errors = (y_pred - y_target)**2
        qf_loss = self.qf_criterion(y_pred, y_target)

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('QF Outputs', q_output),
            ('Bellman Errors', bellman_errors),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
            ('Unregularized QF Loss', qf_loss),
            ('QF Loss', qf_loss),
        ])
