import numpy as np
import torch
from railrl.data_management.path_builder import PathBuilder
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm


class HER(TorchRLAlgorithm):
    """
    Note: this assumes the env will sample the goal when reset() is called,
    i.e. use a "silent" env.

    Hindsight Experience Replay

    This is a template class that should be the first sub-class, i.e.[

    ```
    class HerDdpg(HER, DDPG):
    ```

    and not

    ```
    class HerDdpg(DDPG, HER):
    ```

    Or if you really want to make DDPG the first subclass, do alternatively:
    ```
    class HerDdpg(DDPG, HER):
        def get_batch(self):
            return HER.get_batch(self)
    ```
    for each function defined below.
    """

    def __init__(
            self,
            observation_key=None,
            desired_goal_key=None,
    ):
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key

    def init_rollout_function(self):
        from railrl.samplers.rollout_functions \
                import multitask_rollout, create_rollout_function
        self.rollout_function = create_rollout_function(
            multitask_rollout,
            **dict(
                observation_key=self.observation_key,
                desired_goal_key=self.desired_goal_key
            )
        )

    def _start_new_rollout(self, terminal=True, previous_rollout_last_ob=None):
        self.exploration_policy.reset()
        # Note: we assume we're using a silent env.
        o = self.training_env.reset()
        self._rollout_goal = self.training_env.get_goal()
        return o

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
            goal=None,
    ):
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
            goals=goal if goal is not None else self._rollout_goal,
        )

    def _handle_path(self, path):
        """
        Naive implementation: just loop through each transition.
        :param path:
        :return:
        """
        for (
                ob,
                action,
                reward,
                next_ob,
                goal,
                terminal,
                agent_info,
                env_info
        ) in zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["goals"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        ):
            self._handle_step(
                ob,
                action,
                reward,
                next_ob,
                terminal,
                agent_info=agent_info,
                env_info=env_info,
                goal=goal,
            )
        self._handle_rollout_ending()


    def get_batch(self):
        batch = super().get_batch()
        obs = batch['observations']
        next_obs = batch['next_observations']
        goals = batch['resampled_goals']
        batch['observations'] = torch.cat((
            obs,
            goals
        ), dim=1)
        batch['next_observations'] = torch.cat((
            next_obs,
            goals
        ), dim=1)
        return batch
        # Assume that images are normalized in get_batch rather than in the
        # replay buffer to save memory. Everything starting with 'image' is
        # assumed to be an image.
        for key, val in batch.items():
            if key.startswith('image'):
                batch[key] = normalize_image(val)
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
        goal = self._rollout_goal
        if self.observation_key:
            observation = observation[self.observation_key]
        if self.desired_goal_key:
            goal = self._rollout_goal[self.desired_goal_key]
        new_obs = np.hstack((observation, goal))
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
        return self.rollout_function(
            self.env,
            self.policy,
            self.max_path_length,
            animated=self.render_during_eval
        )

