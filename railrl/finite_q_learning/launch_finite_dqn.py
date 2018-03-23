"""
Try out this finite-horizon q learning.
"""

import gym
import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.finite_q_learning.discrete_q_learning import FiniteDiscreteQLearning
from railrl.launchers.launcher_util import setup_logger, run_experiment
from railrl.torch.networks import Mlp
import railrl.misc.hyperparameter as hyp


def experiment(variant):
    env = gym.make('CartPole-v0')

    qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
    )
    algorithm = FiniteDiscreteQLearning(
        env,
        qf,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 3
    mode = 'ec2'
    exp_prefix = 'cartpole-finite-sweep-H'

    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=50,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=50,
            discount=1,
            random_action_prob=0.2,
            save_environment=False,  # Can't serialize CartPole for some reason
        ),
    )

    search_space = {
        'algo_kwargs.max_path_length': [20, 30, 40, 50],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
            )
