import unittest

import numpy as np

from railrl.data_management.subtraj_replay_buffer import SubtrajReplayBuffer
from railrl.testing.tf_test_case import TFTestCase
from railrl.testing.stub_classes import StubEnv


class TestSubtrajReplayBuffer(TFTestCase):
    def test_size_add_none(self):
        env = StubEnv()
        buff = SubtrajReplayBuffer(
            100,
            env,
            2,
        )
        self.assertEqual(buff.num_steps_can_sample(return_all=True), 0)

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
        self.assertEqual(buff.num_steps_can_sample(return_all=True), 0)

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
        self.assertEqual(buff.num_steps_can_sample(return_all=True), 1)

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
        self.assertEqual(buff.num_steps_can_sample(return_all=True), 2)

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
        self.assertEqual(buff.num_steps_can_sample(return_all=True), 2)

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
        self.assertEqual(buff.num_steps_can_sample(return_all=True), 2)

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
        self.assertEqual(buff.num_steps_can_sample(return_all=True), 2)

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
        self.assertEqual(buff.num_steps_can_sample(return_all=True), 8)

    def test_random_subtraj_shape(self):
        env = StubEnv()
        buff = SubtrajReplayBuffer(
            100,
            env,
            2,
        )
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.terminate_episode(observation)
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

    def test_num_can_sample_validation(self):
        env = StubEnv()
        buff = SubtrajReplayBuffer(
            100,
            env,
            2,
            save_period=1,
        )
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.num_steps_can_sample(validation=False), 0)
        self.assertEqual(buff.num_steps_can_sample(validation=True), 1)

    def test_validation_splitter(self):
        def create_random_generator():
            numbers = [0.1, 0.9, 0.1, 0.1]
            i = 0
            def random_generator():
                nonlocal i
                number = numbers[i]
                i += 1
                return number
            return random_generator

        env = StubEnv()
        buff = SubtrajReplayBuffer(
            100,
            env,
            2,
            save_period=2,
            random_generator=create_random_generator(),
        )
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.terminate_episode(observation)
        action = np.array([[-0.5]])
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.terminate_episode(observation)
        action = np.array([[1]])
        buff.add_sample(observation, action, 1, False)
        buff.add_sample(observation, action, 1, False)
        buff.terminate_episode(observation)
        train_trajs = buff.get_valid_subtrajectories(validation=False)
        validation_trajs = buff.get_valid_subtrajectories(validation=True)
        train_actions = train_trajs['actions']
        valid_actions = validation_trajs['actions']
        self.assertEqual(len(valid_actions), 2)
        self.assertEqual(len(train_actions), 1)
        self.assertNpEqual(valid_actions[0], np.array([[0.5], [0.5]]))
        self.assertNpEqual(valid_actions[1], np.array([[1], [1]]))
        self.assertNpEqual(train_actions[0], np.array([[-0.5], [-0.5]]))


if __name__ == '__main__':
    unittest.main()
