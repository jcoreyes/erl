import unittest
import numpy as np

from railrl.envs.multitask.reacher_env import (
    XyMultitaskReacherEnv,
    XyMultitaskSimpleStateReacherEnv,
    GoalStateSimpleStateReacherEnv,
)
from railrl.testing.np_test_case import NPTestCase


class TestMultitaskReacherEnv(NPTestCase):
    def test_compute_rewards(self):
        env = XyMultitaskReacherEnv()
        for _ in range(10):
            obs = env.reset()
            for _ in range(10):
                action = np.random.uniform(-1, 1, 2)
                next_obs, reward, done, info_dict = env.step(action)
                estimated_reward = env.compute_rewards(
                    None,
                    None,
                    np.expand_dims(obs, 0),
                    np.expand_dims(env.goal, 0),
                )
                distance = np.array([np.linalg.norm(obs[-3:])])
                true_reward = np.array([info_dict['reward_dist']])
                self.assertNpAlmostEqual(-distance, true_reward)
                self.assertNpAlmostEqual(estimated_reward, true_reward)
                obs = next_obs


class TestSimpleReacherEnv(NPTestCase):
    def test_compute_rewards(self):
        env = XyMultitaskSimpleStateReacherEnv()
        for _ in range(10):
            obs = env.reset()
            for _ in range(10):
                action = np.random.uniform(-1, 1, 2)
                next_obs, reward, done, info_dict = env.step(action)
                estimated_reward = env.compute_rewards(
                    None,
                    None,
                    np.expand_dims(obs, 0),
                    np.expand_dims(env.goal, 0),
                )
                true_reward = np.array([info_dict['reward_dist']])
                self.assertNpAlmostEqual(estimated_reward, true_reward)
                obs = next_obs


class TestGoalStateReacherEnv(NPTestCase):
    def test_compute_rewards(self):
        env = GoalStateSimpleStateReacherEnv()
        for _ in range(10):
            obs = env.reset()
            goal = np.expand_dims(obs, 0)
            for _ in range(10):
                action = np.random.uniform(-1, 1, 2)
                next_obs, *_ = env.step(action)
                next_obs = np.expand_dims(next_obs, 0)
                estimated_reward = env.compute_rewards(
                    None,
                    None,
                    next_obs,
                    goal,
                )
                true_reward = -np.linalg.norm(next_obs - goal, axis=1)
                self.assertNpAlmostEqual(estimated_reward, true_reward)


if __name__ == '__main__':
    unittest.main()
