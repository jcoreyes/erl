import unittest

try:
    import numpy as np
    import ConfigSpace
    import fanova
    from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter


    # class TestfANOVAtoyData(unittest.TestCase):
    #     def test_with_toy_data(self):
    #         # self.X = np.random.uniform(-1, 1, size=(10, 2))
    #         # X = np.zeros((10, 2))
    #         X = np.array([
    #             [1],
    #             [2],
    #         ])
    #         y = np.zeros(10)
    #         # y = np.random.rand(10)
    #
    #         cfs = ConfigSpace.ConfigurationSpace()
    #
    #         # f1 = UniformFloatHyperparameter('x1', -999, 999)
    #         # f2 = UniformFloatHyperparameter('x2', -999, 999)
    #         f2 = CategoricalHyperparameter('x2', [0, 1, 2])
    #
    #         # cfs.add_hyperparameter(f1)
    #         cfs.add_hyperparameter(f2)
    #
    #         f = fanova.fANOVA(X, y, cfs)
    #
    #
    # if __name__ == '__main__':
        # unittest.main()
        # self.X = np.random.uniform(-1, 1, size=(10, 2))
        # X = np.zeros((10, 2))
    X = np.array([
        [0, 0],
        [0.1, 1],
        [0.4, 1],
        [0.5, 1],
        [0.6, 1],
        [0.65, 1],
        [1, 1],
    ])
    y = np.random.rand(len(X))

    config_space = ConfigSpace.ConfigurationSpace()
    # config_space.add_hyperparameter(CategoricalHyperparameter('f', [0, 1, 2]))
    config_space.add_hyperparameter(UniformFloatHyperparameter(
        'f', X[:, 0].min(), X[:, 0].max())
    )
    config_space.add_hyperparameter(CategoricalHyperparameter(
        'f2', [0, 1])
    )
    f = fanova.fANOVA(X, y, config_space)

    import sys

    import os
    import pickle
    import tempfile
    import unittest



    class TestfANOVAtoyData(unittest.TestCase):
        def setUp(self):
            self.X = np.loadtxt('/home/vitchyr/tmp_features.csv', delimiter=',')
            self.y = np.loadtxt('/home/vitchyr/tmp_responses.csv', delimiter=',')

            self.cfs = cfs.ConfigurationSpace()

            self.X[:, 0] = np.random.uniform(1, 10, size=len(self.y))
            f1 = cfs.UniformFloatHyperparameter(
                'x1', np.min(self.X[:, 0]), np.max(self.X[:, 0])
            )
            f2 = cfs.CategoricalHyperparameter('x2', [0, 1, 2])

            self.cfs.add_hyperparameter(f1)
            self.cfs.add_hyperparameter(f2)

        def tearDown(self):
            self.X = None
            self.y = None

        def test_with_toy_data(self):
            self.y = self.y[:30]
            f = fanova.fANOVA(self.X, self.y, self.cfs, bootstrapping=True,
                              n_trees=1, seed=5, max_features=1)

            f.the_forest.save_latex_representation('/tmp/fanova_')
            print("=" * 80)
            print(f.the_forest.all_split_values())
            print("total variances", f.the_forest.get_trees_total_variances())
            print(f.quantify_importance([0, 1]))
            print(f.trees_total_variance)
except ImportError:
    print("Skipping test since fanova is not installed")


if __name__ == '__main__':
    unittest.main()
