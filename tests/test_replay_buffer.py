import unittest

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer


class TestSimpleReplayBuffer(unittest.TestCase):

    def test_num_steps_can_sample(self):
        buffer = SimpleReplayBuffer(10000, 1, 1)
        buffer.add_sample(1, 1, 1, False, 1)
        buffer.add_sample(1, 1, 1, True, 1)
        buffer.terminate_episode()
        buffer.add_sample(1, 1, 1, False, 1)
        self.assertEqual(buffer.num_steps_can_sample(), 3)


if __name__ == '__main__':
    unittest.main()
