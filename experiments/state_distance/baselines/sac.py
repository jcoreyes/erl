import random

import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.envs.multitask.half_cheetah import GoalXVelHalfCheetah, \
    GoalXPosHalfCheetah
from railrl.envs.multitask.multitask_env import MultitaskToFlatEnv
from railrl.envs.multitask.pusher3d import MultitaskPusher3DEnv
from railrl.envs.multitask.reacher_7dof import (
    Reacher7DofXyzGoalState,
)
from railrl.envs.multitask.walker2d_env import Walker2DTargetXPos
from railrl.envs.wrappers import normalize_box
from railrl.launchers.launcher_util import run_experiment
from railrl.sac.policies import TanhGaussianPolicy
from railrl.sac.sac import SoftActorCritic
from railrl.torch.networks import FlattenMlp
import railrl.misc.hyperparameter as hyp


def experiment(variant):
    env = variant['env_class']()
    env = normalize_box(env)
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
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-state-distance-sac-baseline"

    n_seeds = 1
    mode = "ec2"
    exp_prefix = "tdm-dense-cheetah"

    num_epochs = 100
    num_steps_per_epoch = 10000
    num_steps_per_eval = 1000
    max_path_length = 50

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
        net_size=300,
        version="SAC",
        algorithm="SAC",
    )
    search_space = {
        'env_class': {
            # Reacher7DofXyzGoalState,
            GoalXVelHalfCheetah,
            # Walker2DTargetXPos,
            # GoalXPosHalfCheetah,
            # MultitaskPusher3DEnv,
        },
        'multitask': [False, True],
        'algo_params.reward_scale': [
            1000, 100, 10, 1,
        ],
        'algo_params.replay_buffer_size': [
            int(1e6),
        ],
        'algo_params.num_updates_per_env_step': [
            1,
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
