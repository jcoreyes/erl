import numpy as np
from railrl.data_management.her_replay_buffer import SimpleHerReplayBuffer, \
    RelabelingReplayBuffer
from railrl.torch.her.her import HER
from railrl.torch.sac.twin_sac import TwinSAC


class HerTwinSac(HER, TwinSAC):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert isinstance(
            self.replay_buffer, SimpleHerReplayBuffer
        ) or isinstance(
            self.replay_buffer, RelabelingReplayBuffer
        )

    def get_eval_action(self, observation, goal):
        new_obs = np.hstack((observation, goal))
        return self.policy.get_action(new_obs, deterministic=True)
