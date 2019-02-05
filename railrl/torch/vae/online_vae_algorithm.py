from railrl.core import logger
from railrl.data_management.shared_obs_dict_replay_buffer \
    import SharedObsDictRelabelingBuffer
import railrl.torch.vae.vae_schedules as vae_schedules
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm
import railrl.torch.pytorch_util as ptu
from torch.multiprocessing import Process, Pipe
from threading import Thread
import numpy as np


class OnlineVaeAlgorithm(TorchRLAlgorithm):

    def __init__(
            self,
            vae,
            vae_trainer,
            vae_save_period=1,
            vae_training_schedule=vae_schedules.never_train,
            oracle_data=False,
            parallel_vae_train=True,
            vae_min_num_steps_before_training=0,
            uniform_dataset=None,
    ):
        self.vae = vae
        self.vae_trainer = vae_trainer
        self.vae_trainer.model = self.vae
        self.vae_save_period = vae_save_period
        self.vae_training_schedule = vae_training_schedule
        self.epoch = 0
        self.oracle_data = oracle_data

        self.vae_training_process = None
        self.process_vae_update_thread = None
        self.parallel_vae_train = parallel_vae_train
        self.vae_min_num_steps_before_training = vae_min_num_steps_before_training
        self.uniform_dataset = uniform_dataset

    def _post_epoch(self, epoch):
        super()._post_epoch(epoch)
        if self.parallel_vae_train and self.vae_training_process is None:
            self.init_vae_training_subproces()

        should_train, amount_to_train = self.vae_training_schedule(epoch)
        if self.replay_buffer._prioritize_vae_samples:
            self.log_priority_weights()
        rl_start_epoch = int(self.min_num_steps_before_training / self.num_env_steps_per_epoch)
        if should_train \
                and self.replay_buffer.num_steps_can_sample() >= self.vae_min_num_steps_before_training \
                or epoch >= (rl_start_epoch - 1):
            if self.parallel_vae_train:
                assert self.vae_training_process.is_alive()
                # Make sure the last vae update has finished before starting
                # another one
                if self.process_vae_update_thread is not None:
                    self.process_vae_update_thread.join()
                self.process_vae_update_thread = Thread(
                    target=OnlineVaeAlgorithm.process_vae_update_thread,
                    args=(self, ptu.device)
                )
                self.process_vae_update_thread.start()
                self.vae_conn_pipe.send((amount_to_train, epoch))
            else:
                self.vae.train()
                _train_vae(
                    self.vae_trainer,
                    self.replay_buffer,
                    epoch,
                    amount_to_train
                )
                self.vae.eval()
                self.replay_buffer.refresh_latents(epoch)
                _test_vae(
                    self.vae_trainer,
                    self.epoch,
                    self.replay_buffer,
                    vae_save_period=self.vae_save_period,
                    uniform_dataset=self.uniform_dataset,
                )
        # very hacky
        self.epoch = epoch + 1

    def log_priority_weights(self):
        vae_sample_priorities = self.replay_buffer._vae_sample_priorities[:self.replay_buffer_size]
        vae_sample_probs = self.replay_buffer._vae_sample_probs
        if self.replay_buffer._vae_sample_probs is None:
            stats = create_stats_ordered_dict(
                'VAE Sample Weights',
                np.zeros(self.replay_buffer._size),
            )
        else:
            stats = create_stats_ordered_dict(
                'VAE Sample Weights',
                vae_sample_probs,
            )
        for key, value in stats.items():
            logger.record_tabular(key, value)

    def reset_vae(self):
        self.vae.init_weights(self.vae.init_w)

    @property
    def networks(self):
        return [self.vae]

    def update_epoch_snapshot(self, snapshot):
        snapshot.update(vae=self.vae)

    def cleanup(self):
        if self.parallel_vae_train:
            self.vae_conn_pipe.close()
            self.vae_training_process.terminate()

    def init_vae_training_subproces(self):
        assert isinstance(self.replay_buffer, SharedObsDictRelabelingBuffer)

        self.vae_conn_pipe, process_pipe = Pipe()
        self.vae_training_process = Process(
            target=subprocess_train_vae_loop,
            args=(
                process_pipe,
                self.vae,
                self.vae.state_dict(),
                self.replay_buffer,
                self.replay_buffer.get_mp_info(),
                ptu.device,
            )
        )
        self.vae_training_process.start()
        self.vae_conn_pipe.send(self.vae_trainer)

    def process_vae_update_thread(self, device):
        self.vae.__setstate__(self.vae_conn_pipe.recv())
        self.vae.to(device)
        _test_vae(
            self.vae_trainer,
            self.epoch,
            self.replay_buffer,
            vae_save_period=self.vae_save_period,
            uniform_dataset=self.uniform_dataset,
        )

    def evaluate(self, epoch, eval_paths=None):
        for k, v in self.vae_trainer.vae_logger_stats_for_rl.items():
            logger.record_tabular(k, v)
        super().evaluate(epoch, eval_paths=eval_paths)


def _train_vae(vae_trainer, replay_buffer, epoch, batches=50, oracle_data=False):
    batch_sampler = replay_buffer.random_vae_training_data
    if oracle_data:
        batch_sampler = None
    vae_trainer.train_epoch(
        epoch,
        sample_batch=batch_sampler,
        batches=batches,
        from_rl=True,
    )
    replay_buffer.train_dynamics_model(batches=batches)


def _test_vae(vae_trainer, epoch, replay_buffer, vae_save_period=1, uniform_dataset=None):
    save_imgs = epoch % vae_save_period == 0
    log_fit_skew_stats = replay_buffer._prioritize_vae_samples and uniform_dataset is not None
    if uniform_dataset is not None:
        replay_buffer.log_loss_under_uniform(uniform_dataset, vae_trainer.batch_size)
    vae_trainer.test_epoch(
        epoch,
        from_rl=True,
        save_reconstruction=save_imgs,
    )
    if save_imgs:
        vae_trainer.dump_samples(epoch)
        if log_fit_skew_stats:
            replay_buffer.dump_best_reconstruction(epoch)
            replay_buffer.dump_worst_reconstruction(epoch)
            replay_buffer.dump_sampling_histogram(epoch, batch_size=vae_trainer.batch_size)
        if uniform_dataset is not None:
            replay_buffer.dump_uniform_imgs_and_reconstructions(dataset=uniform_dataset, epoch=epoch)


def subprocess_train_vae_loop(
        conn_pipe,
        vae,
        vae_params,
        replay_buffer,
        mp_info,
        device,
):
    """
    The observations and next_observations of the replay buffer are stored in
    shared memory. This loop waits until the parent signals to start vae
    training, trains and sends the vae back, and then refreshes the latents.
    Refreshing latents in the subprocess reflects in the main process as well
    since the latents are in shared memory. Since this is does asynchronously,
    it is possible for the main process to see half the latents updated and half
    not.
    """
    ptu.device = device
    vae_trainer = conn_pipe.recv()
    vae.load_state_dict(vae_params)
    vae.to(device)
    vae_trainer.set_vae(vae)
    replay_buffer.init_from_mp_info(mp_info)
    replay_buffer.env.vae = vae
    while True:
        amount_to_train, epoch = conn_pipe.recv()
        vae.train()
        _train_vae(vae_trainer, replay_buffer, epoch, amount_to_train)
        vae.eval()
        conn_pipe.send(vae_trainer.model.__getstate__())
        replay_buffer.refresh_latents(epoch)
