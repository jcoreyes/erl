import random

import numpy as np

import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.mujoco.discrete_swimmer import DiscreteSwimmerEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.dqn.dqn import DQN
from rlkit.torch.networks import Mlp


def experiment(variant):
    env = DiscreteSwimmerEnv(**variant['env_params'])

    qf = Mlp(
        input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
        **variant['qf_kwargs']
    )
    algorithm = DQN(
        env,
        qf=qf,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 2
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=200,
            discount=0.99,
            epsilon=0.5,
            tau=0.001,
        ),
        env_params=dict(
        ),
        qf_kwargs=dict(
            hidden_sizes=[32, 32],
        )
    )
    search_space = {
        # 'env_params.num_bins': [3, 5, 10],
        # 'env_params.reward_position': [False, True],
        # 'algo_params.tau': [0.01, 0.001],
        # 'algo_params.reward_scale': [0.1, 1, 10],
        # 'algo_params.epsilon': [0.1, 0.5],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                seed=seed,
                # exp_prefix="dqn-swimmer-sweep",
                # mode='ec2',
                # use_gpu=False,
                exp_prefix="dev-dqn-swimmer",
                mode='local',
                use_gpu=True,
                variant=variant,
            )
