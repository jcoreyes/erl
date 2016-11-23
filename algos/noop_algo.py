from rllab.algos.base import RLAlgorithm
from rllab.misc import logger
from rllab.misc import special
from rllab.misc.overrides import overrides
from rllab.sampler import parallel_sampler
import numpy as np


class NoOpAlgo(RLAlgorithm):
    """
    Algorithm that doesn't do any training. Used for benchmarking.
    """

    def __init__(
            self,
            env,
            policy,
            exploration_strategy,
            batch_size=32,
            n_epochs=200,
            epoch_length=1000,
            discount=0.99,
            max_path_length=250,
            n_eval_samples=10000):
        """
        :param env: Environment
        :param policy: Policy
        :param exploration_strategy: ExplorationStrategy
        :param batch_size: Minibatch size for training
        :param n_epochs: Number of epoch
        :param epoch_length: Number of time steps per epoch
        :param discount: Discount factor for the MDP
        :param max_path_length: Maximum path length
        :param n_eval_samples: Number of time steps to take for evaluation.
        :return:
        """
        self.env = env
        self.policy = policy
        self.exploration_strategy = exploration_strategy
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.discount = discount
        self.max_path_length = max_path_length
        self.n_eval_samples = n_eval_samples

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)

    @overrides
    def train(self):
        self.start_worker()
        for epoch in range(self.n_epochs):
            logger.push_prefix('Epoch #%d | ' % epoch)
            logger.log("Skipping training for NOOP algorithm.")
            self.evaluate(epoch)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

    def evaluate(self, epoch):
        paths = parallel_sampler.sample_paths(
            policy_params=self.policy.get_param_values(),
            max_samples=self.n_eval_samples,
            max_path_length=self.max_path_length,
        )
        average_discounted_return = np.mean(
            [special.discount_return(path["rewards"], self.discount)
             for path in paths]
        )

        returns = [sum(path["rewards"]) for path in paths]
        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('AverageReturn',
                              np.mean(returns))
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
