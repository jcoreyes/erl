import numpy as np
from railrl.data_management.simple_replay_buffer import SimpleReplayBuffer
from rllab.misc.overrides import overrides


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_pool_size,
            env,
            **kwargs
    ):
        super().__init__(
            max_pool_size=max_pool_size,
            observation_dim=env.observation_space.flat_dim,
            action_dim=env.action_space.flat_dim,
            **kwargs
        )
        self._env = env

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
            flat_action = np.zeros(self._env.action_space.flat_dim)
        else:
            flat_action = self._env.action_space.flatten(action)
        flat_obs = self._env.observation_space.flatten(observation)
        super()._add_sample(
            flat_obs,
            flat_action,
            reward,
            terminal,
            initial,
        )

    def random_batch(self, batch_size, flatten=False):
        batch = super().random_batch(batch_size)

        if flatten:
            return batch

        actions = batch['actions']
        unflat_actions = [self._env.action_space.unflatten(a) for a in actions]
        batch['actions'] = unflat_actions

        obs = batch['observations']
        unflat_obs = [self._env.observation_space.unflatten(o) for o in obs]
        batch['observations'] = unflat_obs

        next_obs = batch['next_observations']
        unflat_next_obs = [self._env.observation_space.unflatten(o) for o in
                           next_obs]
        batch['next_observations'] = unflat_next_obs

        return batch
