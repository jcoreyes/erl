import railrl.torch.vae.vae_schedules as vae_schedules
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm
from railrl.torch.vae.conv_vae import ConvVAE
import railrl.torch.pytorch_util as ptu

class OnlineVaeAlgorithm(TorchRLAlgorithm):

    def __init__(
        self,
        vae,
        vae_trainer,
        vae_save_period=1,
        vae_training_schedule=vae_schedules.never_train,
        oracle_data=False,
        hard_restart_period=1000000,
    ):
        self.vae = vae
        self.vae_trainer = vae_trainer
        self.vae_trainer.model = self.vae
        self.vae_save_period = vae_save_period
        self.vae_training_schedule = vae_training_schedule
        self.epoch = 0
        self.hard_restart_period = hard_restart_period
        self.oracle_data = oracle_data

    def _post_epoch(self, epoch):
        super()._post_epoch(epoch)
        should_train, amount_to_train = self.vae_training_schedule(epoch)
        print(should_train, amount_to_train)
        if epoch % self.hard_restart_period == 0 and epoch != 0:
            self.reset_vae()
            should_train = True
            amount_to_train = 1000
        if should_train:
            self.vae.train()
            self._train_vae(epoch, amount_to_train)
            self.vae.eval()
            self.replay_buffer.refresh_latents(epoch)
        self._test_vae(epoch)
        # very hacky
        self.epoch = epoch + 1

    def reset_vae(self):
        self.vae.init_weights(self.vae.init_w)

    def _post_step(self, step):
        pass

    def _train_vae(self, epoch, batches=50):
        batch_sampler = None
        if not self.oracle_data:
            batch_sampler = self.replay_buffer.random_vae_training_data
        self.vae_trainer.train_epoch(
            epoch,
            sample_batch=batch_sampler,
            batches=batches,
            from_rl=True,
        )
        import time
        cur = time.time()
        self.replay_buffer.train_dynamics_model(batches=batches)
        print(time.time() - cur)

    def _test_vae(self, epoch):
        # batch_sampler = self.replay_buffer.random_vae_training_data
        self.vae_trainer.test_epoch(epoch, from_rl=True)
        if epoch % self.vae_save_period == 0:
            self.vae_trainer.dump_samples(epoch)

    @property
    def networks(self):
        return [self.vae]


    def update_epoch_snapshot(self, snapshot):
        snapshot.update(
            vae=self.vae,
        )

