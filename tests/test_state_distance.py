import unittest
import numpy as np

from railrl.algos.state_distance.state_distance_q_learning import \
    multitask_rollout
from railrl.policies.state_distance import UniversalPolicy
from railrl.testing.stub_classes import StubEnv


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
            decrement_discount=False,
        )
        self.assertTrue(np.all(path['terminals'] == False))
        self.assertTrue(len(path['terminals']) == 100)

    def test_decrement_discount(self):
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
            decrement_discount=True,
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
            decrement_discount=True,
            cycle_tau=True,
        )
        self.assertEqual(
            list(path['taus']),
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
            decrement_discount=True,
            cycle_tau=False,
        )
        self.assertEqual(
            list(path['taus']),
            [5, 4, 3, 2, 1, 0, 0, 0, 0, 0]
        )


class StubUniversalPolicy(UniversalPolicy):
    def set_discount(self, discount):
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
