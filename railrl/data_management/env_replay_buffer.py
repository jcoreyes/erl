import numpy as np
from railrl.data_management.simple_replay_buffer import SimpleReplayBuffer
from railrl.envs.wrappers import convert_gym_space


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            flatten=False,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        :param flatten: Return flatten action, obs, and next_obs when
        returning samples.
        """
        self._ob_space = convert_gym_space(env.observation_space)
        self._action_space = convert_gym_space(env.action_space)
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=self._ob_space.flat_dim,
            action_dim=self._action_space.flat_dim,
        )
        self._env = env
        self.flatten = flatten

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        """

        :param observation: Unflattened observation.
        :param action: Unflattened actions.
        :param reward: number
        :param next_observation: Unflattened next observation.
        :param terminal: Boolean
        """
        flat_action = self._action_space.flatten(action)
        flat_ob = self._ob_space.flatten(observation)
        flat_next_ob = self._ob_space.flatten(next_observation)
        super().add_sample(
            observation=flat_ob,
            action=flat_action,
            reward=reward,
            next_observation=flat_next_ob,
            terminal=terminal,
        )

    def random_batch(self, batch_size):
        batch = super().random_batch(batch_size)

        if self.flatten:
            return batch

        actions = batch['actions']
        unflat_actions = [self._action_space.unflatten(a) for a in actions]
        batch['actions'] = np.array(unflat_actions)

        obs = batch['observations']
        unflat_obs = [self._ob_space.unflatten(o) for o in obs]
        batch['observations'] = np.array(unflat_obs)

        next_obs = batch['next_observations']
        unflat_next_obs = [self._ob_space.unflatten(o) for o in next_obs]
        batch['next_observations'] = np.array(unflat_next_obs)

        return batch
