import unittest

from rlkit.exploration_strategies.product_strategy import ProductStrategy
from rlkit.testing.np_test_case import NPTestCase
from rlkit.testing.stub_classes import AddEs, StubPolicy


class TestProductStrategy(NPTestCase):
    def test_product_strategy_from_policy(self):
        es1 = AddEs(1)
        es2 = AddEs(2)
        policy = StubPolicy((1, 2))
        es = ProductStrategy([es1, es2])
        action, _ = es.get_action(None, None, policy)
        self.assertEqual(action, (2, 4))

    def test_product_strategy_from_raw_action(self):
        es1 = AddEs(1)
        es2 = AddEs(2)
        es = ProductStrategy([es1, es2])
        action = es.get_action_from_raw_action((1, 2))
        self.assertEqual(action, (2, 4))

if __name__ == '__main__':
    unittest.main()
