from railrl.torch.her.her_twin_sac import HerTwinSAC
from railrl.torch.vae.online_vae_algorithm import OnlineVaeAlgorithm
import railrl.torch.vae.vae_schedules as vae_schedules
from railrl.data_management.online_vae_replay_buffer \
        import OnlineVaeRelabelingBuffer

class OnlineVaeHerTwinSac(OnlineVaeAlgorithm, HerTwinSAC):

    def __init__(
        self,
        online_vae_algo_kwargs,
        algo_kwargs,
    ):
        OnlineVaeAlgorithm.__init__(
            self,
            **online_vae_algo_kwargs,
        )
        HerTwinSAC.__init__(self, **algo_kwargs)

        assert isinstance(self.replay_buffer, OnlineVaeRelabelingBuffer)

    @property
    def networks(self):
        return HerTwinSAC.networks.fget(self) + \
               OnlineVaeAlgorithm.networks.fget(self)
