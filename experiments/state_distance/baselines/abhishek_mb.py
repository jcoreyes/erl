import random

import numpy as np
import tensorflow as tf

import railrl.misc.hyperparameter as hyp
from railrl.envs.multitask.her_pusher_env import Pusher2DEnv, \
    pusher2d_cost_fn
from railrl.envs.multitask.her_reacher_7dof_env import Reacher7Dof, \
    reacher7dof_cost_fn
from railrl.launchers.launcher_util import run_experiment


def experiment(variant):
    from cheetah_env import HalfCheetahEnvNew
    from cost_functions import cheetah_cost_fn, \
        hopper_cost_fn, \
        swimmer_cost_fn
    from hopper_env import HopperEnvNew
    from main_solution import train_dagger
    from rllab.misc import logger
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
        if env_name_or_class == Pusher2DEnv:
            cost_fn = pusher2d_cost_fn
        elif env_name_or_class == Reacher7Dof:
            cost_fn = reacher7dof_cost_fn
        else:
            raise NotImplementedError

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
    run_mode = "none"
    snapshot_mode = "last"

    n_seeds = 5
    mode = "ec2"
    exp_prefix = "abhishek-mb-baseline-pusher-again-shaped"
    # snapshot_mode = "gap_and_last"
    snapshot_gap = 10

    # Data collection
    num_steps_per_epoch = 1000
    max_path_length = 100
    num_epochs = 100
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
        ),
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_path', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    use_gpu = True
    if mode != "local":
        use_gpu = False

    search_space = {
        'env_name_or_class': [
            Pusher2DEnv,
            # Reacher7Dof,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for i in range(n_seeds):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            seed = random.randint(0, 999999)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=0,
                use_gpu=use_gpu,
                snapshot_mode=snapshot_mode,
                snapshot_gap=snapshot_gap,
            )
