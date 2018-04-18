import numpy as np
from railrl.data_management.her_replay_buffer import SimpleHerReplayBuffer
from railrl.torch.her.her import HER
from railrl.torch.sac.sac import SoftActorCritic


class HerSac(HER, SoftActorCritic):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert isinstance(self.replay_buffer, SimpleHerReplayBuffer)

    def get_eval_action(self, observation, goal):
        new_obs = np.hstack((observation, goal))
        return self.policy.get_action(new_obs, deterministic=True)
