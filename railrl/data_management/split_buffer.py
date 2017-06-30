import random

from railrl.data_management.replay_buffer import ReplayBuffer


class SplitReplayBuffer(ReplayBuffer):
    """
    Split the data into a training and validation set.
    """
    def __init__(
            self,
            train_replay_buffer: ReplayBuffer,
            validation_replay_buffer: ReplayBuffer,
            fraction_paths_in_train,
    ):
        self.train_replay_buffer = train_replay_buffer
        self.validation_replay_buffer = validation_replay_buffer
        self.fraction_paths_in_train = fraction_paths_in_train
        self.replay_buffer = self.train_replay_buffer

    def sample_replay_buffer(self):
        if random.random() <= self.fraction_paths_in_train:
            return self.train_replay_buffer
        else:
            return self.validation_replay_buffer

    def add_sample(self, *args, **kwargs):
        self.replay_buffer.add_sample(*args, **kwargs)

    def num_steps_can_sample(self):
        return min(
            self.train_replay_buffer.num_steps_can_sample(),
            self.validation_replay_buffer.num_steps_can_sample(),
        )

    def terminate_episode(self, *args, **kwargs):
        self.replay_buffer.terminate_episode(*args, **kwargs)
        self.replay_buffer = self.sample_replay_buffer()

    def random_batch(self, batch_size, training=True, **kwargs):
        if training:
            return self.train_replay_buffer.random_batch(batch_size, **kwargs)
        else:
            return self.validation_replay_buffer.random_batch(batch_size, **kwargs)
