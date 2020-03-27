import numpy as np
from railrl.envs.contextual import ContextualEnv
from railrl.policies.base import Policy
from railrl.samplers.data_collector import MdpPathCollector


class ContextualPathCollector(MdpPathCollector):
    def __init__(
            self,
            env: ContextualEnv,
            policy: Policy,
            max_num_epoch_paths_saved=None,
            observation_key='observation',
            context_key='context',
            render=False,
            render_kwargs=None,
    ):
        def obs_processor(o):
            return np.hstack((o[observation_key], o[context_key]))
        super().__init__(
            env, policy, max_num_epoch_paths_saved, render, render_kwargs,
            preprocess_obs_for_policy_fn=obs_processor
        )
        self._observation_key = observation_key
        self._context_key = context_key

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
            context_key=self._context_key,
        )
        return snapshot
