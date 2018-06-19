from railrl.torch.vae.online_vae_algorithm import OnlineVaeAlgorithm
import railrl.torch.vae.vae_schedules as vae_schedules
from railrl.data_management.online_vae_replay_buffer \
        import OnlineVaeRelabelingBuffer
from railrl.torch.her.her_joint_algo import HerJointAlgo

class OnlineVaeHerJointAlgo(OnlineVaeAlgorithm, HerJointAlgo):

    def __init__(
        self,
        vae,
        vae_trainer,
        *algo_args,
        vae_save_period=1,
        vae_training_schedule=vae_schedules.never_train,

        **algo_kwargs
    ):
        OnlineVaeAlgorithm.__init__(
            self,
            vae,
            vae_trainer,
            vae_save_period=vae_save_period,
            vae_training_schedule=vae_training_schedule
        )
        HerJointAlgo.__init__(self, *algo_args, **algo_kwargs)

        assert isinstance(self.replay_buffer, OnlineVaeRelabelingBuffer)
