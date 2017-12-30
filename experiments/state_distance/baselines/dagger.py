import random
import numpy as np

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.dagger.controller import MPCController
from railrl.dagger.dagger import Dagger
from railrl.dagger.model import DynamicsModel
from railrl.envs.multitask.ant_env import GoalXYPosAnt
from railrl.envs.multitask.half_cheetah import GoalXVelHalfCheetah, \
    GoalXPosHalfCheetah
from railrl.envs.multitask.multitask_env import MultitaskToFlatEnv, \
    MultitaskEnvToSilentMultitaskEnv
from railrl.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from railrl.envs.multitask.pusher3d import MultitaskPusher3DEnv
from railrl.envs.multitask.reacher_7dof import (
    Reacher7DofXyzGoalState)
from railrl.envs.multitask.walker2d_env import Walker2DTargetXPos
from railrl.envs.wrappers import convert_gym_space, normalize_box
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.data_management.normalizer import TorchFixedNormalizer
from railrl.torch.networks import FlattenMlp


def experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    if variant['multitask']:
        env = MultitaskEnvToSilentMultitaskEnv(env)
    env = normalize_box(
        env,
        **variant['normalize_kwargs']
    )

    observation_dim = int(np.prod(env.observation_space.low.shape))
    action_dim = int(np.prod(env.action_space.low.shape))
    obs_normalizer = TorchFixedNormalizer(observation_dim)
    action_normalizer = TorchFixedNormalizer(action_dim)
    delta_normalizer = TorchFixedNormalizer(observation_dim)
    model = DynamicsModel(
        observation_dim=observation_dim,
        action_dim=action_dim,
        obs_normalizer=obs_normalizer,
        action_normalizer=action_normalizer,
        delta_normalizer=delta_normalizer,
        **variant['model_kwargs']
    )
    mpc_controller = MPCController(
        env,
        model,
        env.cost_fn,
        **variant['mpc_controller_kwargs']
    )
    algo = Dagger(
        env,
        model,
        mpc_controller,
        obs_normalizer=obs_normalizer,
        action_normalizer=action_normalizer,
        delta_normalizer=delta_normalizer,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        algo.cuda()
    algo.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-dagger"

    n_seeds = 1
    mode = "ec2"
    exp_prefix = "normal-gear-ratio-ant-d6"

    num_epochs = 1000
    num_steps_per_epoch = 1000
    num_steps_per_eval = 1000
    max_path_length = 100

    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            collection_mode='online',
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
            num_steps_per_eval=num_steps_per_eval,
            num_updates_per_epoch=10,
            max_path_length=max_path_length,
            learning_rate=1e-3,
            num_updates_per_env_step=1,
            batch_size=512,
            num_paths_for_normalization=20,
        ),
        normalize_kwargs=dict(
            obs_mean=None,
            obs_std=None,
        ),
        mpc_controller_kwargs=dict(
            num_simulated_paths=512,
            mpc_horizon=15,
        ),
        model_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        version="Model-Based-Dagger-online-training",
        algorithm="Model-Based-Dagger",
    )
    search_space = {
        'env_class': [
            # Reacher7DofXyzGoalState,
            # GoalXVelHalfCheetah,
            GoalXYPosAnt,
            # GoalXPosHalfCheetah,
            # CylinderXYPusher2DEnv,
            # MultitaskPusher3DEnv,
            # Walker2DTargetXPos,
        ],
        'multitask': [True],
        'algo_kwargs.num_paths_for_normalization': [20],
        'algo_kwargs.collection_mode': ['online'],
        'env_kwargs.max_distance': [
            6,
        ],
        'env_kwargs.use_low_gear_ratio': [
            False,
        ],
        'algo_kwargs.batch_size': [128],
        'mpc_controller_kwargs.mpc_horizon': [15],
        'algo_kwargs.num_updates_per_env_step': [
            1,
        ],
        'algo_kwargs.max_path_length': [
            50, 100
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            seed = random.randint(0, 999999)
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                seed=seed,
                variant=variant,
                exp_id=exp_id,
            )
