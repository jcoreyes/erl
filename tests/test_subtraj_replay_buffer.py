import unittest

import numpy as np

from rlkit.data_management.subtraj_replay_buffer import SubtrajReplayBuffer
from rlkit.testing.np_test_case import NPTestCase
from rlkit.testing.stub_classes import StubEnv


def create_buffer(subtraj_length):
    env = StubEnv()
    return SubtrajReplayBuffer(
        100,
        env,
        subtraj_length,
    )


class TestSubtrajReplayBuffer(NPTestCase):
    def test_size_add_none(self):
        subtraj_length = 2
        buff = create_buffer(subtraj_length)
        self.assertEqual(buff.num_steps_can_sample(return_all=True), 0)

    def test_size_add_one(self):
        subtraj_length = 2
        buff = create_buffer(subtraj_length)
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.num_steps_can_sample(return_all=True), 0)

    def test_size_enough_for_one_subtraj(self):
        subtraj_length = 2
        buff = create_buffer(subtraj_length)
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.num_steps_can_sample(return_all=True),
                         1 * subtraj_length)

    def test_size_enough_for_two_subtrajs(self):
        subtraj_length = 2
        buff = create_buffer(subtraj_length)
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.num_steps_can_sample(return_all=True),
                         2 * subtraj_length)

        buff = create_buffer(subtraj_length)
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.terminate_episode(observation, True)
        self.assertEqual(buff.num_steps_can_sample(return_all=True),
                         2 * subtraj_length)

    def test_size_after_terminate(self):
        subtraj_length = 2
        buff = create_buffer(subtraj_length)
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.terminate_episode(observation, True)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.num_steps_can_sample(return_all=True),
                         2 * subtraj_length)

    def test_size_after_terminal_true(self):
        subtraj_length = 2
        buff = create_buffer(subtraj_length)
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, True)
        buff.terminate_episode(observation, True)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.num_steps_can_sample(return_all=True),
                         2 * subtraj_length)

    def test_size_add_many(self):
        subtraj_length = 2
        buff = create_buffer(subtraj_length)
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        for _ in range(10):
            buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.num_steps_can_sample(return_all=True),
                         8 * subtraj_length)

    def test_random_subtraj_shape(self):
        subtraj_length = 2
        buff = create_buffer(subtraj_length)
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.terminate_episode(observation, True)
        for _ in range(10):
            buff.add_sample(observation, action, 1, False)
        subtrajs = buff.random_subtrajectories(5)
        self.assertEqual(subtrajs['observations'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['actions'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['next_observations'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['rewards'].shape, (5, 2))
        self.assertEqual(subtrajs['terminals'].shape, (5, 2))

    def test_get_all_valid_subtrajectories(self):
        subtraj_length = 2
        buff = create_buffer(subtraj_length)
        buff.add_sample(np.array([[1]]), np.array([[-1]]), 1, False)
        buff.add_sample(np.array([[2]]), np.array([[-2]]), 1, True)
        buff.terminate_episode(np.array([[0]]), True)
        buff.add_sample(np.array([[3]]), np.array([[-3]]), 1, False)
        buff.add_sample(np.array([[4]]), np.array([[-4]]), 1, False)
        buff.add_sample(np.array([[5]]), np.array([[-5]]), 1, False)
        subtrajs = buff.get_all_valid_subtrajectories()

        self.assertNpEqual(
            subtrajs['observations'],
            np.array([
                [[1], [2]],
                [[3], [4]],
            ])
        )
        self.assertNpEqual(
            subtrajs['actions'],
            np.array([
                [[-1], [-2]],
                [[-3], [-4]],
            ])
        )
        self.assertNpEqual(
            subtrajs['next_observations'],
            np.array([
                [[2], [0]],
                [[4], [5]],
            ])
        )
        self.assertNpEqual(
            subtrajs['rewards'],
            np.array([
                [1, 1],
                [1, 1],
            ])
        )
        self.assertNpEqual(
            subtrajs['terminals'],
            np.array([
                [False, True],
                [False, False],
            ])
        )


if __name__ == '__main__':
    unittest.main()
