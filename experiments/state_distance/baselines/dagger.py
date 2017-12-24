import random

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.dagger.controller import MPCController
from railrl.dagger.dagger import Dagger
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

    observation_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    model = FlattenMlp(
        input_size=int(observation_space.flat_dim) + int(action_space.flat_dim),
        output_size=int(observation_space.flat_dim),
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
        **variant['dagger_kwargs']
    )
    if ptu.gpu_enabled():
        algo.cuda()
    algo.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-dagger"

    # n_seeds = 3
    # mode = "ec2"
    # exp_prefix = "dagger"

    dagger_iters = 10
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
