import numpy as np

import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.multitask.point2d import MultitaskPoint2DEnv
from rlkit.envs.multitask.point2d_uwall import MultitaskPoint2dUWall
from rlkit.torch.mpc.controller import MPCController
from rlkit.torch.mpc.model_trainer import ModelTrainer
from rlkit.torch.mpc.dynamics_model import DynamicsModel
from rlkit.envs.multitask.multitask_env import MultitaskEnvToSilentMultitaskEnv
from rlkit.envs.multitask.reacher_7dof import (
    Reacher7DofFullGoal
)
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer


def experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    if variant['multitask']:
        env = MultitaskEnvToSilentMultitaskEnv(env)
    env = NormalizedBoxEnv(
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
    algo = ModelTrainer(
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
        algo.to(ptu.device)
    algo.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-mpc-neural-networks"

    # n_seeds = 3
    # mode = "ec2"
    # exp_prefix = "reacher-full-mpcnn-save-replay-buffer"

    num_epochs = 100
    num_steps_per_epoch = 100
    num_steps_per_eval = 100
    max_path_length = 20

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
            save_replay_buffer=True,
            replay_buffer_size=30000,
            render=True,
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
            theta=0.5,
            max_sigma=1.0,
            min_sigma=1.0,
        ),
        env_kwargs=dict(),
        version="Model-Based-Dagger",
        algorithm="Model-Based-Dagger",
    )
    search_space = {
        'multitask': [True],
        'env_class': [
            # Reacher7DofFullGoal,
            MultitaskPoint2dUWall,
            # MultitaskPoint2DEnv,
            # GoalXVelHalfCheetah,
            # Reacher7DofXyzGoalState,
            # GoalXYPosAnt,
            # GoalXPosHalfCheetah,
            # GoalXYGymPusherEnv,
            # MultitaskPusher3DEnv,
            # GoalXPosHopper,
            # Reacher7DofXyzPosAndVelGoalState,
            # GoalXYPosAndVelAnt,
            # GoalXYPosAndVelAnt,
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
        # 'env_kwargs.speed_weight': [
        #     None,
        # ],
        # 'env_kwargs.goal_dim_weights': [
        #     (0.1, 0.1, 0.9, 0.9),
        # ],
        # 'env_kwargs.done_threshold': [
        #     0.005,
        # ],
        # 'algo_kwargs.max_path_length': [
        #     max_path_length,
        # ],
        'algo_kwargs.num_updates_per_env_step': [
            1,
        ],
        'mpc_controller_kwargs.mpc_horizon': [10],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                variant=variant,
                exp_id=exp_id,
                use_gpu=True,
            )
