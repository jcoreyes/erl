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
from railrl.envs.multitask.multitask_env import MultitaskToFlatEnv
from railrl.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from railrl.envs.multitask.pusher3d import MultitaskPusher3DEnv
from railrl.envs.multitask.reacher_7dof import (
    Reacher7DofXyzGoalState)
from railrl.envs.wrappers import convert_gym_space, normalize_box
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.data_management.normalizer import TorchFixedNormalizer
from railrl.torch.networks import FlattenMlp


def experiment(variant):
    env_class = variant['env_class']
    env = env_class()
    if variant['multitask']:
        env = MultitaskToFlatEnv(env)
    env = normalize_box(
        env,
        **variant['normalize_kwargs']
    )

    observation_dim = int(np.prod(env.observation_space.low.shape))
    action_dim = int(np.prod(env.action_space.low.shape))
    obs_normalizer = TorchFixedNormalizer(observation_dim)
    action_normalizer = TorchFixedNormalizer(action_dim)
    model = DynamicsModel(
        observation_dim=observation_dim,
        action_dim=action_dim,
        obs_normalizer=obs_normalizer,
        action_normalizer=action_normalizer,
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
        **variant['dagger_kwargs']
    )
    if ptu.gpu_enabled():
        algo.cuda()
    algo.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-dagger"

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "model-based-reacher-multitask-fixed-2"

    dagger_iters = 100
    dynamics_iters = 60
    num_paths_dagger = 20

    max_path_length = 50
    num_epochs = dagger_iters
    num_steps_per_epoch = num_paths_dagger * max_path_length
    num_steps_per_eval = num_paths_dagger * max_path_length
    num_updates_per_epoch = dynamics_iters

    # noinspection PyTypeChecker
    variant = dict(
        dagger_kwargs=dict(
            collection_mode='batch',
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
            num_steps_per_eval=num_steps_per_eval,
            num_updates_per_epoch=num_updates_per_epoch,
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
        version="Dagger",
        algorithm="Dagger",
    )
    search_space = {
        'env_class': [
            Reacher7DofXyzGoalState,
            # GoalXVelHalfCheetah,
            # GoalXYPosAnt,
            # GoalXPosHalfCheetah,
            # CylinderXYPusher2DEnv,
            # MultitaskPusher3DEnv,
        ],
        'multitask': [True, False],
        'dagger_kwargs.num_paths_for_normalization': [20, 0],
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
