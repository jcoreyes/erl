import random

import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.multitask.hopper_env import GoalXPosHopper
from rlkit.envs.multitask.multitask_env import MultitaskToFlatEnv
from rlkit.envs.multitask.point2d_uwall import MultitaskPoint2dUWall
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp
import rlkit.misc.hyperparameter as hyp


def experiment(variant):
    env = NormalizedBoxEnv(variant['env_class'](**variant['env_kwargs']))
    if variant['multitask']:
        env = MultitaskToFlatEnv(env)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    algorithm = SoftActorCritic(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-state-distance-sac-baseline"

    # n_seeds = 1
    # mode = "ec2"
    # exp_prefix = "try-hopper-again"

    num_epochs = 100
    num_steps_per_epoch = 1000
    num_steps_per_eval = 1000
    max_path_length = 100

    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
            num_steps_per_eval=num_steps_per_eval,
            max_path_length=max_path_length,
            batch_size=128,
            discount=0.99,

            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        env_kwargs=dict(),
        net_size=300,
        version="SAC",
        algorithm="SAC",
    )
    search_space = {
        'env_class': {
            # Reacher7DofXyzGoalState,
            # GoalXVelHalfCheetah,
            # CylinderXYPusher2DEnv,
            # GoalXYPosAnt,
            # GoalXPosHopper,
            MultitaskPoint2dUWall,
            # Walker2DTargetXPos,
            # GoalXPosHalfCheetah,
            # MultitaskPusher3DEnv,
        },
        # 'env_kwargs.max_distance': [
        #     0.5, 2
        # ],
        # 'env_kwargs.action_penalty': [
        #     1e-3, 0,
        # ],
        'multitask': [True],
        'algo_params.reward_scale': [
            100,
        ],
        'algo_params.replay_buffer_size': [
            int(1e6),
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            seed = random.randint(0, 999999)
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                seed=seed,
                variant=variant,
                exp_id=exp_id,
            )
