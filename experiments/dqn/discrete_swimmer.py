import random

import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.envs.mujoco.discrete_swimmer import DiscreteSwimmerEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.networks.base import Mlp
from railrl.torch.dqn import DQN


def experiment(variant):
    env = DiscreteSwimmerEnv()

    qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
    )
    algorithm = DQN(
        env,
        qf=qf,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
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
            # render=True,
        ),
    )
    seed = random.randint(0, 999999)
    run_experiment(
        experiment,
        exp_prefix="dqn-swimmer",
        seed=seed,
        mode='local',
        variant=variant,
        use_gpu=True,
    )
