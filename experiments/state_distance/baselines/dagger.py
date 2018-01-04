import random
import numpy as np

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.dagger.controller import MPCController
from railrl.dagger.dagger import Dagger
from railrl.dagger.model import DynamicsModel
from railrl.envs.multitask.ant_env import GoalXYPosAnt, GoalXYPosAndVelAnt
from railrl.envs.multitask.half_cheetah import GoalXVelHalfCheetah, \
    GoalXPosHalfCheetah
from railrl.envs.multitask.hopper_env import GoalXPosHopper
from railrl.envs.multitask.multitask_env import MultitaskToFlatEnv, \
    MultitaskEnvToSilentMultitaskEnv
from railrl.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from railrl.envs.multitask.pusher3d import MultitaskPusher3DEnv
from railrl.envs.multitask.pusher3d_gym import GoalXYGymPusherEnv
from railrl.envs.multitask.reacher_7dof import (
    Reacher7DofXyzGoalState, Reacher7DofXyzPosAndVelGoalState)
from railrl.envs.multitask.walker2d_env import Walker2DTargetXPos
from railrl.envs.wrappers import convert_gym_space, normalize_box
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
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
    es = OUStrategy(
        action_space=env.action_space,
        **variant['ou_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=mpc_controller,
    )
    algo = Dagger(
        env,
        model,
        mpc_controller,
        exploration_policy=exploration_policy,
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
    mode = "local_docker"
    exp_prefix = "dev-dagger-2"

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "final-ant-pos-and-vel"

    num_epochs = 500
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
            batch_size=128,
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
        ou_kwargs=dict(
            theta=0.1,
            max_sigma=0.1,
        ),
        env_kwargs=dict(),
        version="Model-Based-Dagger",
        algorithm="Model-Based-Dagger",
    )
    search_space = {
        'multitask': [True],
        'env_class': [
            # GoalXVelHalfCheetah,
            # Reacher7DofXyzGoalState,
            # GoalXYPosAnt,
            # GoalXPosHalfCheetah,
            # GoalXYGymPusherEnv,
            # MultitaskPusher3DEnv,
            # GoalXPosHopper,
            # Reacher7DofXyzPosAndVelGoalState,
            GoalXYPosAndVelAnt,
            # CylinderXYPusher2DEnv,
            # Walker2DTargetXPos,
        ],
        # 'env_kwargs.max_distance': [
        #     6,
        # ],
        # 'env_kwargs.min_distance': [
        #     3,
        # ],
        # 'env_kwargs.reward_coefs': [
        #     (1, 0, 0),
        #     (0.5, 0.375, 0.125),
        # ],
        # 'env_kwargs.norm_order': [
        #     1,
        #     2,
        # ],
        # 'env_kwargs.max_speed': [
        #     0.05,
        # ],
        'env_kwargs.speed_weight': [
            None,
        ],
        'env_kwargs.goal_dim_weights': [
            (0.1, 0.1, 0.9, 0.9),
        ],
        # 'env_kwargs.done_threshold': [
        #     0.005,
        # ],
        # 'algo_kwargs.max_path_length': [
        #     max_path_length,
        # ],
        'algo_kwargs.num_updates_per_env_step': [
            1, 5, 10
        ],
        'algo_kwargs.num_paths_for_normalization': [20],
        'ou_kawrgs.max_sigma': [0.1, 0],
        'mpc_controller_kwargs.mpc_horizon': [15],
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
