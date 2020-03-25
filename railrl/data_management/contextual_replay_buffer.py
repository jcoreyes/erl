import abc
from typing import Any

import numpy as np

from railrl.core.distribution import Distribution
from railrl.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from railrl.envs.contextual import ContextualRewardFn


class SampleContextFromObsDictFn(object, metaclass=abc.ABCMeta):
    """Interface definer, but you can also just pass in a function."""

    @abc.abstractmethod
    def __call__(self, obs_dict) -> Any:
        pass


class SelectKeyFn(SampleContextFromObsDictFn):
    def __init__(self, key):
        self._key = key

    def __call__(self, obs_dict) -> Any:
        return obs_dict[self._key]


class ContextualRelabelingReplayBuffer(ObsDictReplayBuffer):
    """
    Save goals from the same trajectory into the replay buffer.
    Only add_path is implemented.

    Implementation details:
     - Every sample from [0, self._size] will be valid.
     - Observation and next observation are saved separately. It's a memory
       inefficient to save the observations twice, but it makes the code
       *much* easier since you no longer have to worry about termination
       conditions.
    """

    def __init__(
            self,
            max_size,
            env,
            context_key,
            sample_context_from_obs_dict_fn: SampleContextFromObsDictFn,
            reward_fn: ContextualRewardFn,
            context_distribution: Distribution,
            fraction_future_context,
            fraction_distribution_context,
            ob_keys_to_save=None,
            observation_key_for_reward_fn=None,
            post_process_batch_fn=None,
            **kwargs
    ):
        ob_keys_to_save = ob_keys_to_save or []
        if context_key not in ob_keys_to_save:
            ob_keys_to_save.append(context_key)
        super().__init__(
            max_size, env, ob_keys_to_save=ob_keys_to_save, **kwargs)
        if (
            fraction_distribution_context < 0
            or fraction_future_context < 0
            or (fraction_future_context
                + fraction_distribution_context) > 1
        ):
            raise ValueError("Invalid fractions: {} and {}".format(
                fraction_future_context,
                fraction_distribution_context,
            ))
        if observation_key_for_reward_fn is None:
            observation_key_for_reward_fn = self.observation_key
        self._context_key = context_key
        self._context_distribution = context_distribution
        self._sample_context_from_obs_dict_fn = sample_context_from_obs_dict_fn
        self._reward_fn = reward_fn
        self._fraction_future_context = fraction_future_context
        self._fraction_distribution_context = (
            fraction_distribution_context
        )
        self._observation_key_for_reward_fn = observation_key_for_reward_fn
        self._post_process_batch_fn = post_process_batch_fn

    def random_batch(self, batch_size):
        num_future_contexts = int(batch_size * self._fraction_future_context)
        num_distrib_contexts = int(
            batch_size * self._fraction_distribution_context)
        num_rollout_contexts = (
                batch_size - num_future_contexts - num_distrib_contexts
        )
        indices = self._sample_indices(batch_size)
        obs_dict = self._batch_obs_dict(indices)
        next_obs_dict = self._batch_next_obs_dict(indices)
        contexts = [
            next_obs_dict[self._context_key][:num_rollout_contexts]
        ]

        if num_distrib_contexts > 0:
            sampled_contexts = self._context_distribution.sample(
                num_distrib_contexts)
            contexts.append(sampled_contexts)

        if num_future_contexts > 0:
            start_state_indices = indices[-num_future_contexts:]
            future_contexts = self._get_future_contexts(start_state_indices)
            contexts.append(future_contexts)

        actions = self._actions[indices]
        new_contexts = np.concatenate(contexts)
        new_rewards = self._reward_fn(
            obs_dict[self._observation_key_for_reward_fn],
            actions,
            next_obs_dict[self._observation_key_for_reward_fn],
            new_contexts,
        )
        if len(new_rewards.shape) == 1:
            new_rewards = new_rewards.reshape(-1, 1)
        batch = {
            'observations': obs_dict[self.observation_key],
            'actions': actions,
            'rewards': new_rewards,
            'terminals': self._terminals[indices],
            'next_observations': next_obs_dict[self.observation_key],
            'indices': np.array(indices).reshape(-1, 1),
            'contexts': new_contexts,
        }
        if self._post_process_batch_fn:
            batch = self._post_process_batch_fn(batch)
        return batch

    def _get_future_contexts(self, start_state_indices):
        future_obs_idxs = self._get_future_obs_indices(start_state_indices)
        future_obs_dict = self._batch_next_obs_dict(future_obs_idxs)
        return self._sample_context_from_obs_dict_fn(future_obs_dict)

    def _get_future_obs_indices(self, start_state_indices):
        future_obs_idxs = []
        for i in start_state_indices:
            possible_future_obs_idxs = self._idx_to_future_obs_idx[i]
            # This is generally faster than random.choice. Makes you wonder what
            # random.choice is doing
            num_options = len(possible_future_obs_idxs)
            next_obs_i = int(np.random.randint(0, num_options))
            future_obs_idxs.append(possible_future_obs_idxs[next_obs_i])
        future_obs_idxs = np.array(future_obs_idxs)
        return future_obs_idxs


