from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.torch.algos.util import np_to_pytorch_batch
from railrl.torch.rl_algorithm import RLAlgorithm


class TorchRLAlgorithm(RLAlgorithm):
    def __init__(self, *args, replay_buffer=None, **kwargs):
        super().__init__(*args, **kwargs)
        if replay_buffer is None:
            self.replay_buffer = SplitReplayBuffer(
                EnvReplayBuffer(
                    self.replay_buffer_size,
                    self.env,
                    flatten=True,
                ),
                EnvReplayBuffer(
                    self.replay_buffer_size,
                    self.env,
                    flatten=True,
                ),
                fraction_paths_in_train=0.8,
            )
        else:
            self.replay_buffer = replay_buffer

    def get_batch(self, training=True):
        replay_buffer = self.replay_buffer.get_replay_buffer(training)
        sample_size = min(
            replay_buffer.num_steps_can_sample(),
            self.batch_size
        )
        batch = replay_buffer.random_batch(sample_size)
        return np_to_pytorch_batch(batch)
