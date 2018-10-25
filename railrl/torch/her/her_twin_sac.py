import numpy as np
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from railrl.samplers.rollout_functions import (
    create_rollout_function,
    multitask_rollout,
)
from railrl.torch.her.her import HER
from railrl.torch.sac.twin_sac import TwinSAC


class HerTwinSAC(HER, TwinSAC):
    def __init__(
            self,
            *args,
            twin_sac_kwargs,
            her_kwargs,
            base_kwargs,
            **kwargs
    ):
        HER.__init__(
            self,
            **her_kwargs,
        )
        TwinSAC.__init__(self, *args, **kwargs, **twin_sac_kwargs, **base_kwargs)
        assert isinstance(
            self.replay_buffer, ObsDictRelabelingBuffer
        )
        self.eval_rollout_function = create_rollout_function(
            multitask_rollout,
            observation_key=self.observation_key,
            desired_goal_key=self.desired_goal_key,
            get_action_kwargs=dict(deterministic=True),
        )

    def get_eval_action(self, observation, goal):
        if self.observation_key:
            observation = observation[self.observation_key]
        if self.desired_goal_key:
            goal = goal[self.desired_goal_key]
        new_obs = np.hstack((observation, goal))
        return self.policy.get_action(new_obs, deterministic=True)
