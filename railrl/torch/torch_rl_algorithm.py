from railrl.torch.algos.util import np_to_pytorch_batch
from railrl.torch.rl_algorithm import RLAlgorithm


class TorchRLAlgorithm(RLAlgorithm):
    def get_batch(self, training=True):
        replay_buffer = self.replay_buffer.get_replay_buffer(training)
        sample_size = min(
            replay_buffer.num_steps_can_sample(),
            self.batch_size
        )
        batch = replay_buffer.random_batch(sample_size)
        return np_to_pytorch_batch(batch)
