import railrl.torch.vae.vae_schedules as vae_schedules
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm

class OnlineVaeAlgorithm(TorchRLAlgorithm):

    def __init__(
        self,
        vae,
        vae_trainer,
        vae_save_period=1,
        vae_training_schedule=vae_schedules.never_train,
    ):
        self.vae = vae
        self.vae_trainer = vae_trainer
        self.vae_trainer.model = self.vae
        self.vae_save_period = vae_save_period
        self.vae_training_schedule = vae_training_schedule
        self.epoch = 0

    def _post_epoch(self, epoch):
        should_train, amount_to_train = self.vae_training_schedule(epoch)
        if should_train:
            self.vae.train()
            self._train_vae(epoch, amount_to_train)
            self.vae.eval()
            self.replay_buffer.refresh_latents(epoch)
        self._test_vae(epoch)
        # very hacky
        self.epoch = epoch + 1

    def _post_step(self, step):
        pass

    def _train_vae(self, epoch, batches=50):
        batch_sampler = self.replay_buffer.random_vae_training_data
        self.vae_trainer.train_epoch(
            epoch,
            sample_batch=batch_sampler,
            batches=batches,
            from_rl=True,
        )

    def _test_vae(self, epoch):
#        batch_sampler = self.replay_buffer.random_vae_training_data
        self.vae_trainer.test_epoch(epoch, from_rl=True)
        if epoch % self.vae_save_period == 0:
            self.vae_trainer.dump_samples(epoch)

