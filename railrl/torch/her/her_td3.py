from railrl.data_management.her_replay_buffer import SimpleHerReplayBuffer, \
    RelabelingReplayBuffer
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from railrl.torch.her.her import HER
from railrl.torch.td3.td3 import TD3


class HerTd3(HER, TD3):
    def __init__(
            self,
            *args,
            observation_key=None,
            desired_goal_key=None,
            **kwargs
    ):
        HER.__init__(
            self,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
        )
        TD3.__init__(self, *args, **kwargs)
        assert isinstance(
            self.replay_buffer, SimpleHerReplayBuffer
        ) or isinstance(
            self.replay_buffer, RelabelingReplayBuffer
        ) or isinstance(
            self.replay_buffer, ObsDictRelabelingBuffer
        )
