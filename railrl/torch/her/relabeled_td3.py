
from railrl.data_management.her_replay_buffer import RelabelingReplayBuffer
from railrl.torch.her.her import HER
from railrl.torch.td3.td3 import TD3


class RelabeledTd3(HER, TD3):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        # assert isinstance(self.replay_buffer, HerReplayBuffer)
