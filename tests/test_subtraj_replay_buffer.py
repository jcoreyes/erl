import unittest

import numpy as np

from railrl.data_management.subtraj_replay_buffer import SubtrajReplayBuffer
from railrl.testing.tf_test_case import TFTestCase
from rllab.spaces.box import Box


class TestSubtrajReplayBuffer(TFTestCase):
    def test_size_add_none(self):
        env = StubEnv()
        buff = SubtrajReplayBuffer(
            100,
            env,
            2,
        )
        self.assertEqual(buff.num_can_sample, 0)

    def test_size_add_one(self):
        env = StubEnv()
        buff = SubtrajReplayBuffer(
            100,
            env,
            2,
        )
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.num_can_sample, 0)

    def test_size_enough_for_one_subtraj(self):
        env = StubEnv()
        buff = SubtrajReplayBuffer(
            100,
            env,
            2,
        )
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.num_can_sample, 1)

    def test_size_enough_for_two_subtrajs(self):
        env = StubEnv()
        buff = SubtrajReplayBuffer(
            100,
            env,
            2,
        )
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.num_can_sample, 2)

        buff = SubtrajReplayBuffer(
            100,
            env,
            2,
        )
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.terminate_episode(observation)
        self.assertEqual(buff.num_can_sample, 2)

    def test_size_after_terminate(self):
        env = StubEnv()
        buff = SubtrajReplayBuffer(
            100,
            env,
            2,
        )
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.terminate_episode(observation)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.num_can_sample, 2)

    def test_size_after_terminal_true(self):
        env = StubEnv()
        buff = SubtrajReplayBuffer(
            100,
            env,
            2,
        )
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, True)
        buff.terminate_episode(observation)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.num_can_sample, 2)

    def test_size_add_many(self):
        env = StubEnv()
        buff = SubtrajReplayBuffer(
            100,
            env,
            2,
        )
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        for _ in range(10):
            buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.num_can_sample, 8)

    def test_random_subtraj_shape(self):
        env = StubEnv()
        buff = SubtrajReplayBuffer(
            100,
            env,
            2,
        )
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        for _ in range(10):
            buff.add_sample(observation, action, 1, False)
        subtrajs = buff.random_subtrajectories(5)
        self.assertEqual(subtrajs['observations'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['actions'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['next_observations'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['rewards'].shape, (5, 2))
        self.assertEqual(subtrajs['terminals'].shape, (5, 2))

    def test_get_all_valid_subtrajectories(self):
        env = StubEnv()
        buff = SubtrajReplayBuffer(
            100,
            env,
            2,
        )
        buff.add_sample(np.array([[1]]), np.array([[-1]]), 1, False)
        buff.add_sample(np.array([[2]]), np.array([[-2]]), 1, True)
        buff.terminate_episode(np.array([[0]]))
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


class StubEnv(object):
    def __init__(self):
        low = np.array([0.])
        high = np.array([1.])
        self.action_space = Box(low, high)
        self.observation_space = Box(low, high)


if __name__ == '__main__':
    unittest.main()
