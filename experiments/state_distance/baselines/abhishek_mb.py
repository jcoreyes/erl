import random

import numpy as np
import tensorflow as tf

import rlkit.misc.hyperparameter as hyp
from rlkit.envs.multitask.ant_env import GoalXYPosAnt
from rlkit.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from rlkit.envs.multitask.her_half_cheetah import HalfCheetah, \
    half_cheetah_cost_fn
from rlkit.envs.multitask.her_pusher_env import Pusher2DEnv, \
    pusher2d_cost_fn
from rlkit.envs.multitask.her_reacher_7dof_env import Reacher7Dof, \
    reacher7dof_cost_fn
from rlkit.envs.multitask.reacher_7dof import (
    Reacher7DofXyzGoalState,
)
from rlkit.envs.multitask.pusher3d import MultitaskPusher3DEnv
from rlkit.envs.multitask.multitask_env import MultitaskToFlatEnv
from rlkit.launchers.launcher_util import run_experiment


def experiment(variant):
    from cheetah_env import HalfCheetahEnvNew
    from cost_functions import cheetah_cost_fn, \
        hopper_cost_fn, \
        swimmer_cost_fn
    from hopper_env import HopperEnvNew
    from main_solution import train_dagger
    from rlkit.core import logger
    from swimmer_env import SwimmerEnvNew
    env_name_or_class = variant['env_name_or_class']

    if type(env_name_or_class) == str:
        if 'cheetah' in str.lower(env_name_or_class):
            env = HalfCheetahEnvNew()
            cost_fn = cheetah_cost_fn
        elif 'hopper' in str.lower(env_name_or_class):
            env = HopperEnvNew()
            cost_fn = hopper_cost_fn
        elif 'swimmer' in str.lower(env_name_or_class):
            env = SwimmerEnvNew()
            cost_fn = swimmer_cost_fn
        else:
            raise NotImplementedError
    else:
        env = env_name_or_class()
        from rlkit.envs.wrappers import NormalizedBoxEnv
        env = NormalizedBoxEnv(env)
        if env_name_or_class == Pusher2DEnv:
            cost_fn = pusher2d_cost_fn
        elif env_name_or_class == Reacher7Dof:
            cost_fn = reacher7dof_cost_fn
        elif env_name_or_class == HalfCheetah:
            cost_fn = half_cheetah_cost_fn
        else:
            if variant['multitask']:
                env = MultitaskToFlatEnv(env)
            cost_fn = env.cost_fn

    train_dagger(
        env=env,
        cost_fn=cost_fn,
        logdir=logger.get_snapshot_dir(),
        **variant['dagger_params']
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Experiment meta-params
    parser.add_argument('--exp_name', type=str, default='mb_mpc')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--render', action='store_true')
    # Training args
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--dagger_iters', '-n', type=int, default=10)
    parser.add_argument('--dyn_iters', '-nd', type=int, default=60)
    parser.add_argument('--batch_size', '-b', type=int, default=512)
    # Neural network architecture args
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=300)
    # MPC Controller
    parser.add_argument('--simulated_paths', '-sp', type=int, default=512)
    parser.add_argument('--mpc_horizon', '-m', type=int, default=15)
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-abhishek-mb"

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "model-based-reacher-multitask-fixed-2"

    num_epochs = 100
    num_steps_per_epoch = 1000
    max_path_length = 50

    variant = dict(
        # env='HalfCheetah-v1',
        env_name_or_class='HalfCheetah-v1',
        dagger_params=dict(
            render=args.render,
            learning_rate=args.learning_rate,
            dagger_iters=num_epochs,
            dynamics_iters=args.dyn_iters,
            batch_size=args.batch_size,
            num_paths_random=num_steps_per_epoch // max_path_length,
            num_paths_dagger=num_steps_per_epoch // max_path_length,
            num_simulated_paths=args.simulated_paths,
            env_horizon=max_path_length,
            mpc_horizon=args.mpc_horizon,
            n_layers=2,
            size=300,
            activation=tf.nn.relu,
            output_activation=None,
            normalize=True,
        ),
        multitask=True,
        version="Model-Based - Abhishek",
        algorithm="Model-Based",
    )

    use_gpu = True
    if mode != "local":
        use_gpu = False

    search_space = {
        'env_name_or_class': [
            Reacher7DofXyzGoalState,
            # MultitaskPusher3DEnv,
            # GoalXYPosAnt,
            # CylinderXYPusher2DEnv,
        ],
        'multitask': [True],
        'dagger_params.normalize': [True, False],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if variant['env_name_or_class'] == CylinderXYPusher2DEnv:
            max_path_length = 100
            variant['dagger_params']['num_paths_random'] = (
                num_steps_per_epoch // max_path_length
            )
            variant['dagger_params']['num_paths_dagger'] = (
                num_steps_per_epoch // max_path_length
            )
            variant['dagger_params']['env_horizon'] = (
                max_path_length
            )
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
