from functools import partial

from railrl.envs.contextual import ContextualEnv
from railrl.policies.base import Policy
from railrl.samplers.data_collector import MdpPathCollector
from railrl.samplers.rollout_functions import contextual_rollout


class ContextualPathCollector(MdpPathCollector):
    def __init__(
            self,
            env: ContextualEnv,
            policy: Policy,
            max_num_epoch_paths_saved=None,
            observation_key='observation',
            context_keys='context',
            render=False,
            render_kwargs=None,
    ):
        rollout_fn = partial(
            contextual_rollout,
            context_keys=context_keys,
            observation_key=observation_key,
        )
        super().__init__(
            env, policy, max_num_epoch_paths_saved, render, render_kwargs,
            rollout_fn=rollout_fn,
        )
        self._observation_key = observation_key
        self._context_keys = context_keys

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
            context_keys=self._context_keys,
        )
        return snapshot
