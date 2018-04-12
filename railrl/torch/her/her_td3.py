import numpy as np
import torch
from railrl.data_management.her_replay_buffer import SimpleHerReplayBuffer
from railrl.data_management.path_builder import PathBuilder
from railrl.torch.td3.td3 import TD3


class HerTd3(TD3):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert isinstance(self.replay_buffer, SimpleHerReplayBuffer)

    def _start_new_rollout(self, terminal=True, previous_rollout_last_ob=None):
        self.exploration_policy.reset()
        self._rollout_goal = self.env.sample_goal_for_rollout()
        self.training_env.set_goal(self._rollout_goal)
        return self.training_env.reset()

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
            goals=self._rollout_goal,
        )

    def get_batch(self):
        batch = super().get_batch()
        obs = batch['observations']
        next_obs = batch['next_observations']
        goals = batch['goals']
        batch['observations'] = torch.cat((
            obs,
            goals
        ), dim=1)
        batch['next_observations'] = torch.cat((
            next_obs,
            goals
        ), dim=1)
        return batch

    def _handle_rollout_ending(self):
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self.replay_buffer.add_path(path)
            self._exploration_paths.append(path)
            self._current_path_builder = PathBuilder()

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        new_obs = np.hstack((observation, self._rollout_goal))
        return self.exploration_policy.get_action(new_obs)

    def get_eval_paths(self):
        paths = []
        n_steps_total = 0
        while n_steps_total <= self.num_steps_per_eval:
            path = self.eval_multitask_rollout()
            paths.append(path)
            n_steps_total += len(path['observations'])
        return paths

    def eval_multitask_rollout(self):
        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        next_observations = []
        o = self.env.reset()
        path_length = 0
        goal = self.env.sample_goal_for_rollout()
        self.env.set_goal(goal)
        while path_length < self.max_path_length:
            new_obs = np.hstack((o, goal))
            a, agent_info = self.policy.get_action(new_obs)
            next_o, r, d, env_info = self.env.step(a)
            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            next_observations.append(next_o)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            if d:
                break
            o = next_o

        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        return dict(
            observations=observations,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
            goals=np.repeat(goal[None], path_length, 0),
        )
