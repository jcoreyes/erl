from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from railrl.torch.her.her import HER
from railrl.demos.behavior_clone import BehaviorClone


class HerBC(HER, BehaviorClone):
    def __init__(
            self,
            *args,
            td3_kwargs,
            her_kwargs,
            base_kwargs,
            **kwargs
    ):
        HER.__init__(
            self,
            **her_kwargs,
        )
        BehaviorClone.__init__(self, *args, **kwargs, **td3_kwargs, **base_kwargs)
        assert isinstance(
            self.replay_buffer, ObsDictRelabelingBuffer
        )
