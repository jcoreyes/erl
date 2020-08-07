import unittest

import numpy as np

from rlkit.state_distance.policies import UniversalPolicy
from rlkit.state_distance.rollout_util import multitask_rollout
from rlkit.testing.stub_classes import StubEnv


class TestMultitaskRollout(unittest.TestCase):
    def test_multitask_rollout_length(self):
        env = StubMultitaskEnv()
        policy = StubUniversalPolicy()
        goal = None
        discount = 1
        path = multitask_rollout(
            env,
            policy,
            goal,
            discount,
            max_path_length=100,
            animated=False,
            decrement_tau=False,
        )
        self.assertTrue(np.all(path['terminals'] == False))
        self.assertTrue(len(path['terminals']) == 100)

    def test_decrement_tau(self):
        env = StubMultitaskEnv()
        policy = StubUniversalPolicy()
        goal = None
        tau = 10
        path = multitask_rollout(
            env,
            policy,
            goal,
            tau,
            max_path_length=tau,
            animated=False,
            decrement_tau=True,
        )
        self.assertTrue(np.all(path['terminals'] == False))
        self.assertTrue(len(path['terminals']) == tau)

    def test_tau_cycles(self):
        env = StubMultitaskEnv()
        policy = StubUniversalPolicy()
        goal = None
        tau = 5
        path = multitask_rollout(
            env,
            policy,
            goal,
            tau,
            max_path_length=10,
            animated=False,
            decrement_tau=True,
            cycle_tau=True,
        )
        self.assertEqual(
            list(path['num_steps_left']),
            [5, 4, 3, 2, 1, 0, 5, 4, 3, 2]
        )

    def test_decrement_tau(self):
        env = StubMultitaskEnv()
        policy = StubUniversalPolicy()
        goal = None
        tau = 5
        path = multitask_rollout(
            env,
            policy,
            goal,
            tau,
            max_path_length=10,
            animated=False,
            decrement_tau=True,
            cycle_tau=False,
        )
        self.assertEqual(
            list(path['num_steps_left']),
            [5, 4, 3, 2, 1, 0, 0, 0, 0, 0]
        )


class StubUniversalPolicy(UniversalPolicy):
    def set_tau(self, tau):
        pass

    def set_goal(self, goal_np):
        pass

    def get_action(self, obs):
        return 0, {}


class StubMultitaskEnv(StubEnv):
    def set_goal(self, goal):
        pass


if __name__ == '__main__':
    unittest.main()
