import unittest

from railrl.data_management.simple_replay_buffer import SimpleReplayBuffer


class TestSimpleReplayBuffer(unittest.TestCase):

    def test_valid_start_indices_episode_done(self):
        buffer = SimpleReplayBuffer(10000, 1, 1)
        buffer.add_sample(1, 1, 1, False)
        buffer.add_sample(1, 1, 1, True)
        buffer.terminate_episode(1, True)
        buffer.add_sample(1, 1, 1, False)
        self.assertEqual(buffer._valid_transition_indices, [0, 1])

    def test_valid_start_indices_episode_not_done(self):
        buffer = SimpleReplayBuffer(10000, 1, 1)
        buffer.add_sample(1, 1, 1, False)
        buffer.add_sample(1, 1, 1, False)
        buffer.terminate_episode(1, False)
        buffer.add_sample(1, 1, 1, False)
        self.assertEqual(buffer._valid_transition_indices, [0, 1])

    def test_valid_start_indices_mix(self):
        buffer = SimpleReplayBuffer(10000, 1, 1)
        buffer.add_sample(1, 1, 1, False)
        buffer.add_sample(1, 1, 1, True)
        buffer.terminate_episode(1, True)
        buffer.add_sample(1, 1, 1, False)
        buffer.add_sample(1, 1, 1, False)
        buffer.terminate_episode(1, False)
        buffer.add_sample(1, 1, 1, False)
        buffer.add_sample(1, 1, 1, False)
        self.assertEqual(buffer._valid_transition_indices,
                         [0, 1, 3, 4, 6])

if __name__ == '__main__':
    unittest.main()
