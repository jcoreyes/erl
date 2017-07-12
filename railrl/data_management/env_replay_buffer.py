import numpy as np
from railrl.data_management.simple_replay_buffer import SimpleReplayBuffer
from railrl.envs.wrappers import convert_gym_space
from rllab.misc.overrides import overrides


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_pool_size,
            env,
            flatten=False,
            **kwargs
    ):
        """
        :param max_pool_size:
        :param env:
        :param flatten: Flatten action, obs, and next_obs when returning samples
        :param kwargs: kwargs to pass to SimpleReplayBuffer constructor.
        """
        self._obs_space = convert_gym_space(env.observation_space)
        self._action_space = convert_gym_space(env.action_space)
        super().__init__(
            max_pool_size=max_pool_size,
            observation_dim=self._obs_space.flat_dim,
            action_dim=self._action_space.flat_dim,
            **kwargs
        )
        self._env = env
        self.flatten = flatten

    @overrides
    def _add_sample(self, observation, action, reward, terminal, initial,
                    **kwargs):
        """

        :param observation: Unflattened observation. If None, will assume to
        be all zeros.
        :param action: Unflattened actions. If None, will assume to be all
        zeros.
        :param reward: int
        :param terminal: Boolean
        :param initial: Boolean
        :return: None
        """
        if action is None:
            flat_action = np.zeros(self._action_space.flat_dim)
        else:
            flat_action = self._action_space.flatten(action)
        flat_obs = self._obs_space.flatten(observation)
        super()._add_sample(
            flat_obs,
            flat_action,
            reward,
            terminal,
            initial,
        )

    def random_batch(self, batch_size):
        batch = super().random_batch(batch_size)

        if self.flatten:
            return batch

        actions = batch['actions']
        unflat_actions = [self._action_space.unflatten(a) for a in actions]
        batch['actions'] = np.array(unflat_actions)

        obs = batch['observations']
        unflat_obs = [self._obs_space.unflatten(o) for o in obs]
        batch['observations'] = np.array(unflat_obs)

        next_obs = batch['next_observations']
        unflat_next_obs = [self._obs_space.unflatten(o) for o in next_obs]
        batch['next_observations'] = np.array(unflat_next_obs)

        return batch
