import abc
from typing import Any, Dict

import numpy as np

from railrl.core.distribution import DictDistribution
from railrl.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from railrl.envs.contextual import ContextualRewardFn
from railrl import pythonplusplus as ppp


class SampleContextFromObsDictFn(object, metaclass=abc.ABCMeta):
    """Interface definer, but you can also just pass in a function.

    This function maps an observation to some context that ``was achieved''.
    """

    @abc.abstractmethod
    def __call__(self, obs: dict) -> Any:
        pass


class RemapKeyFn(SampleContextFromObsDictFn):
    """A simple map that forwards observations to become the context."""
    def __init__(self, context_to_input_key: Dict[str, str]):
        self._context_to_input_key = context_to_input_key

    def __call__(self, obs: dict) -> Any:
        return {
            k: obs[v]
            for k, v in self._context_to_input_key.items()
        }


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
            context_keys,
            observation_keys, # TODO: rename as observation_keys_to_save
            sample_context_from_obs_dict_fn: SampleContextFromObsDictFn,
            reward_fn: ContextualRewardFn,
            context_distribution: DictDistribution,
            fraction_future_context,
            fraction_distribution_context,
            post_process_batch_fn=None,
            observation_key='observation',
            save_data_in_snapshot=False,
            internal_keys=None,
    ):
        ob_keys_to_save = observation_keys + context_keys
        super().__init__(
            max_size,
            env,
            ob_keys_to_save=ob_keys_to_save,
            internal_keys=internal_keys,
            observation_key=observation_key,
            save_data_in_snapshot=save_data_in_snapshot,
        )
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
        self._context_keys = context_keys
        self._context_distribution = context_distribution
        for k in context_keys:
            distribution_keys = set(self._context_distribution.spaces.keys())
            if k not in distribution_keys:
                raise TypeError("All context keys must be in context distribution.")
        self._sample_context_from_obs_dict_fn = sample_context_from_obs_dict_fn
        self._reward_fn = reward_fn
        self._fraction_future_context = fraction_future_context
        self._fraction_distribution_context = (
            fraction_distribution_context
        )
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
        contexts = [{
            k: next_obs_dict[k][:num_rollout_contexts]
            for k in self._context_keys
        }]

        if num_distrib_contexts > 0:
            sampled_contexts = self._context_distribution.sample(
                num_distrib_contexts)
            contexts.append(sampled_contexts)

        if num_future_contexts > 0:
            start_state_indices = indices[-num_future_contexts:]
            future_contexts = self._get_future_contexts(start_state_indices)
            contexts.append(future_contexts)

        actions = self._actions[indices]

        keys = set(contexts[0].keys())
        for c in contexts[1:]:
            if set(c.keys()) != keys:
                raise RuntimeError(
                    "Context distributions don't match. Replay buffer context "
                    "distribution keys={}, other distribution keys={}".format(
                        keys,
                        set(c.keys())
                    )
                )

        def concat(*x):
            return np.concatenate(x, axis=0)
        new_contexts = ppp.treemap(concat, *tuple(contexts),
                                   atomic_type=np.ndarray)
        new_rewards = self._reward_fn(
            obs_dict,
            actions,
            next_obs_dict,
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
            **new_contexts
            # 'contexts': new_contexts,
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


