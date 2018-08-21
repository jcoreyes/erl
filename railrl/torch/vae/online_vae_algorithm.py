import railrl.torch.vae.vae_schedules as vae_schedules
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm
from railrl.torch.vae.conv_vae import ConvVAE
import railrl.torch.pytorch_util as ptu
from railrl.core import logger

class OnlineVaeAlgorithm(TorchRLAlgorithm):

    def __init__(
        self,
        vae,
        vae_trainer,
        vae_save_period=1,
        vae_training_schedule=vae_schedules.never_train,
        oracle_data=False,
    ):
        self.vae = vae
        self.vae_trainer = vae_trainer
        self.vae_trainer.model = self.vae
        self.vae_save_period = vae_save_period
        self.vae_training_schedule = vae_training_schedule
        self.epoch = 0
        self.oracle_data = oracle_data

    def _post_epoch(self, epoch):
        super()._post_epoch(epoch)
        should_train, amount_to_train = self.vae_training_schedule(epoch)
        if should_train:
            self.vae.train()
            self._train_vae(epoch, amount_to_train)
            self.vae.eval()
            self.replay_buffer.refresh_latents(epoch)
        self._test_vae(epoch)

        for log_key, log_val in self.vae_trainer.vae_logger_stats_for_rl.items():
            logger.record_tabular(log_key, log_val)
        # very hacky
        self.epoch = epoch + 1

    def _post_step(self, step):
        pass

    def reset_vae(self):
        self.vae.init_weights(self.vae.init_w)

    def _train_vae(self, epoch, batches=50):
        batch_sampler = self.replay_buffer.random_vae_training_data
        if self.oracle_data:
            batch_sampler = None
        self.vae_trainer.train_epoch(
            epoch,
            sample_batch=batch_sampler,
            batches=batches,
            from_rl=True,
        )
        self.replay_buffer.train_dynamics_model(batches=batches)

    def _test_vae(self, epoch):
        save_imgs = epoch % self.vae_save_period == 0
        self.vae_trainer.test_epoch(
            epoch,
            from_rl=True,
            save_reconstruction=save_imgs,
        )
        if save_imgs:
            self.vae_trainer.dump_samples(epoch)

    @property
    def networks(self):
        return [self.vae]

    def update_epoch_snapshot(self, snapshot):
        snapshot.update(vae=self.vae)

