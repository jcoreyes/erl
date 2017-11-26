import random

import numpy as np

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.envs.multitask.mountain_car_env import MountainCar
from railrl.launchers.launcher_util import run_experiment
from railrl.state_distance.discrete_tdm import DiscreteTDM
from railrl.torch.networks import Mlp


def experiment(variant):
    env = MountainCar()

    qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
    )
    replay_buffer = HerReplayBuffer(
        env=env,
        **variant['her_replay_buffer_params']
    )
    algorithm = DiscreteTDM(
        env,
        qf=qf,
        replay_buffer=replay_buffer,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            base_kwargs=dict(
                num_epochs=500,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                batch_size=128,
                max_path_length=1000,
                discount=0.99,
            ),
            tdm_kwargs=dict(),
            dqn_kwargs=dict(
                epsilon=0.2,
                tau=0.001,
                hard_update_period=1000,
            ),
        ),
        her_replay_buffer_params=dict(
            max_size=int(1E6),
            num_goals_to_sample=4,
        ),
    )
    search_space = {
        'algo_params.dqn_kwargs.use_hard_updates': [True, False],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                exp_prefix="discrete-tdm-mountain-car",
                seed=seed,
                variant=variant,
                exp_id=exp_id,
                # mode='ec2',
                # use_gpu=False,
                mode='local',
                use_gpu=True,
            )
