
from railrl.data_management.her_replay_buffer import SimpleHerReplayBuffer
from railrl.torch.ddpg.ddpg import DDPG
from railrl.torch.her.her import HER


class HerDdpg(HER, DDPG):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert isinstance(self.replay_buffer, SimpleHerReplayBuffer)
