"""Test different rl algorithms."""
import argparse

from algo_launchers import (
    test_my_ddpg,
    test_naf,
    test_convex_naf,
    test_random,
    test_shane_ddpg,
    test_rllab_vpg,
    test_rllab_trpo,
    test_rllab_ddpg,
    test_dqicnn,
    test_quadratic_ddpg,
    test_convex_quadratic_naf,
)
from misc import hyperparameter as hp
from rllab.envs.gym_env import GymEnv

# Seems to do nothing, but this is needed to use local envs.
# import envs

BATCH_SIZE = 128
N_EPOCHS = 100
EPOCH_LENGTH = 10000
EVAL_SAMPLES = 10000
DISCOUNT = 0.99
QF_LEARNING_RATE = 1e-3
POLICY_LEARNING_RATE = 1e-4
BATCH_LEARNING_RATE = 1e-2
SOFT_TARGET_TAU = 1e-2
REPLAY_POOL_SIZE = 1000000
MIN_POOL_SIZE = 10000
SCALE_REWARD = 1.0
QF_WEIGHT_DECAY = 0.00
MAX_PATH_LENGTH = 1000
N_UPDATES_PER_TIME_STEP = 5

# Sweep settings
SWEEP_N_EPOCHS = 50
SWEEP_EPOCH_LENGTH = 10000
SWEEP_EVAL_SAMPLES = 1000
SWEEP_MIN_POOL_SIZE = BATCH_SIZE

# Fast settings
FAST_N_EPOCHS = 100
FAST_EPOCH_LENGTH = 100
FAST_EVAL_SAMPLES = 100
FAST_MIN_POOL_SIZE = 256
FAST_MAX_PATH_LENGTH = 1000

NUM_SEEDS_PER_CONFIG = 3
NUM_HYPERPARAMETER_CONFIGS = 50


def gym_env(name):
    return GymEnv(name,
                  record_video=False,
                  log_dir='/tmp/gym-test',  # Ignore gym log.
                  record_log=False)


def get_algo_settings(algo_name, render=False):
    sweeper = hp.RandomHyperparameterSweeper()
    params = {}
    if algo_name == 'ddpg':
        sweeper = hp.RandomHyperparameterSweeper([
            hp.LogFloatParam("soft_target_tau", 0.005, 0.1),
            hp.LogFloatParam("scale_reward", 10.0, 0.01),
            hp.LogFloatParam("qf_weight_decay", 1e-7, 1e-1),
        ])
        params = get_ddpg_params()
        params['render'] = render
        algorithm_launcher = test_my_ddpg
    elif algo_name == 'shane-ddpg':
        sweeper = hp.RandomHyperparameterSweeper([
            hp.LogFloatParam("soft_target_tau", 0.005, 0.1),
            hp.LogFloatParam("scale_reward", 10.0, 0.01),
            hp.LogFloatParam("qf_weight_decay", 1e-7, 1e-1),
        ])
        params = get_ddpg_params()
        if params['min_pool_size'] <= params['batch_size']:
            params['min_pool_size'] = params['batch_size'] + 1
        algorithm_launcher = test_shane_ddpg
    elif algo_name == 'qddpg':
        sweeper = hp.RandomHyperparameterSweeper([
            hp.LogFloatParam("soft_target_tau", 0.005, 0.1),
            hp.LogFloatParam("scale_reward", 10.0, 0.01),
            hp.LogFloatParam("qf_weight_decay", 1e-7, 1e-1),
            hp.LogFloatParam("qf_learning_rate", 1e-6, 1e-2),
            hp.LogFloatParam("policy_learning_rate", 1e-6, 1e-2),
        ])
        params = get_ddpg_params()
        algorithm_launcher = test_quadratic_ddpg
    elif algo_name == 'cnaf':
        scale_rewards = [100., 10., 1., 0.1, 0.01, 0.001]
        sweeper = hp.RandomHyperparameterSweeper([
            hp.FixedParam("n_epochs", 25),
            hp.FixedParam("epoch_length", 20),
            hp.FixedParam("eval_samples", 20),
            hp.FixedParam("min_pool_size", 20),
            hp.FixedParam("batch_size", 32),
        ])
        global NUM_HYPERPARAMETER_CONFIGS
        NUM_HYPERPARAMETER_CONFIGS = len(scale_rewards)
        params = get_my_naf_params()
        params['render'] = render
        params['optimizer_type'] = 'sgd'
        algorithm_launcher = test_convex_naf
    elif algo_name == 'cqnaf':
        scale_rewards = [100., 10., 1., 0.1, 0.01, 0.001]
        sweeper = hp.RandomHyperparameterSweeper([
            hp.FixedParam("n_epochs", 25),
            hp.FixedParam("epoch_length", 20),
            hp.FixedParam("eval_samples", 20),
            hp.FixedParam("min_pool_size", 20),
            hp.FixedParam("batch_size", 32),
            # hp.LogFloatParam("qf_learning_rate", 1e-7, 1e-1),
            # hp.LogFloatParam("qf_weight_decay", 1e-6, 1e-1),
            # hp.LogFloatParam("soft_target_tau", 0.005, 0.1),
            hp.ListedParam("scale_reward", scale_rewards),
            # hp.LinearFloatParam("discount", .25, 0.99),
        ])
        global NUM_HYPERPARAMETER_CONFIGS
        NUM_HYPERPARAMETER_CONFIGS = len(scale_rewards)
        params = get_my_naf_params()
        params['render'] = render
        params['optimizer_type'] = 'sgd'
        algorithm_launcher = test_convex_quadratic_naf
    elif algo_name == 'naf':
        sweeper = hp.RandomHyperparameterSweeper([
            hp.LogFloatParam("qf_learning_rate", 1e-6, 1e-2),
            hp.LogFloatParam("scale_reward", 10.0, 0.01),
            hp.LogFloatParam("soft_target_tau", 0.001, 0.1),
            hp.LogFloatParam("qf_weight_decay", 1e-6, 1e-1),
            hp.LinearIntParam("n_updates_per_time_step", 1, 10),
        ])
        params = get_my_naf_params()
        params['render'] = render
        algorithm_launcher = test_naf
    elif algo_name == 'dqicnn':
        algorithm_launcher = test_dqicnn
        sweeper = hp.RandomHyperparameterSweeper([
            hp.FixedParam("n_epochs", 25),
            hp.FixedParam("epoch_length", 100),
            hp.FixedParam("eval_samples", 100),
            hp.FixedParam("min_pool_size", 100),
            hp.LogFloatParam("qf_learning_rate", 1e-7, 1e-1),
            hp.LogFloatParam("qf_weight_decay", 1e-6, 1e-1),
            hp.LogFloatParam("soft_target_tau", 0.005, 0.1),
            hp.LogFloatParam("scale_reward", 10.0, 0.01),
        ])
        params = get_my_naf_params()
        params['render'] = render
    elif algo_name == 'random':
        algorithm_launcher = test_random
    elif algo_name == 'rl-vpg':
        algorithm_launcher = test_rllab_vpg
        params = dict(
            batch_size=BATCH_SIZE,
            max_path_length=MAX_PATH_LENGTH,
            n_itr=N_EPOCHS,
            discount=DISCOUNT,
            optimizer_args=dict(
                tf_optimizer_args=dict(
                    learning_rate=BATCH_LEARNING_RATE,
                )
            ),
        )
    elif algo_name == 'rl-trpo':
        algorithm_launcher = test_rllab_trpo
        params = dict(
            batch_size=BATCH_SIZE,
            max_path_length=MAX_PATH_LENGTH,
            n_itr=N_EPOCHS,
            discount=DISCOUNT,
            step_size=BATCH_LEARNING_RATE,
        )
    elif algo_name == 'rl-ddpg':
        algorithm_launcher = test_rllab_ddpg
        params = get_ddpg_params()
        if params['min_pool_size'] <= params['batch_size']:
            params['min_pool_size'] = params['batch_size'] + 1
    else:
        raise Exception("Algo name not recognized: " + algo_name)

    return {
        'sweeper': sweeper,
        'algo_params': params,
        'algorithm_launcher': algorithm_launcher,
    }


def get_ddpg_params():
    return dict(
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        epoch_length=EPOCH_LENGTH,
        eval_samples=EVAL_SAMPLES,
        discount=DISCOUNT,
        policy_learning_rate=POLICY_LEARNING_RATE,
        qf_learning_rate=QF_LEARNING_RATE,
        soft_target_tau=SOFT_TARGET_TAU,
        replay_pool_size=REPLAY_POOL_SIZE,
        min_pool_size=MIN_POOL_SIZE,
        scale_reward=SCALE_REWARD,
        max_path_length=MAX_PATH_LENGTH,
        qf_weight_decay=QF_WEIGHT_DECAY,
    )


def get_my_naf_params():
    return dict(
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        epoch_length=EPOCH_LENGTH,
        eval_samples=EVAL_SAMPLES,
        discount=DISCOUNT,
        qf_learning_rate=QF_LEARNING_RATE,
        soft_target_tau=SOFT_TARGET_TAU,
        replay_pool_size=REPLAY_POOL_SIZE,
        min_pool_size=MIN_POOL_SIZE,
        scale_reward=SCALE_REWARD,
        max_path_length=MAX_PATH_LENGTH,
        qf_weight_decay=QF_WEIGHT_DECAY,
        n_updates_per_time_step=N_UPDATES_PER_TIME_STEP,
    )


def sweep(exp_prefix, env_params, algo_settings):
    sweeper = algo_settings['sweeper']
    test_function = algo_settings['test_function']
    default_params = algo_settings['algo_params']
    for i in range(NUM_HYPERPARAMETER_CONFIGS):
        for seed in range(NUM_SEEDS_PER_CONFIG):
            algo_params = dict(default_params,
                          **sweeper.generate_random_hyperparameters())
            test_function(algo_params, env_params, exp_prefix, seed=seed + 1)


def benchmark(args):
    """
    Benchmark everything!
    """
    name = args.name + "-benchmark"
    env_ids = ['cheetah']
    algo_names = ['qddpg', 'ddpg']
    for env_id in env_ids:
        for seed in range(NUM_SEEDS_PER_CONFIG):
            for algo_name in algo_names:
                algo_settings = get_algo_settings(algo_name, render=False)
                test_function = algo_settings['test_function']
                algo_params = algo_settings['algo_params']
                env_params = {
                    'env_id': env_id,
                    'normalize_env': True,
                }
                test_function(algo_params, env_params, name, seed=seed)


def get_algo_settings_from_args(args):
    return get_algo_settings(args.algo, args.render)


def get_env_params_from_args(args):
    return dict(
        env_id=args.env,
        normalize_env=not args.nonorm,
        gym_name=args.gym
    )

def main():
    env_choices = ['ant', 'cheetah', 'cart', 'point', 'pt', 'reacher',
                   'idp', 'gym']
    algo_choices = ['ddpg', 'naf', 'shane-ddpg', 'random', 'cnaf', 'cqnaf',
                    'rl-vpg', 'rl-trpo', 'rl-ddpg', 'dqicnn', 'qddpg']
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action='store_true',
                        help="Run benchmarks.")
    parser.add_argument("--sweep", action='store_true',
                        help="Sweep _hyperparameters for my DDPG.")
    parser.add_argument("--render", action='store_true',
                        help="Render the environment.")
    parser.add_argument("--env", default='cart',
                        help="Test algo on 'cart' or 'cheetah'.",
                        choices=env_choices)
    parser.add_argument("--gym",
                        help="Gym env name if 'gym' was given as the env")
    parser.add_argument("--name", default='default',
                        help='Experiment prefix')
    parser.add_argument("--fast", action='store_true',
                        help=('Run a quick experiment. Intended for debugging. '
                              'Overrides sweep settings'))
    parser.add_argument("--nonorm", action='store_true',
                        help="Normalize the environment")
    parser.add_argument("--algo", default='ddpg',
                        help='Algo',
                        choices=algo_choices)
    parser.add_argument("--seed", default=1,
                        type=int,
                        help='Seed')
    parser.add_argument("--num_seeds", default=NUM_SEEDS_PER_CONFIG, type=int,
                        help="Run this many seeds, starting with --seed.")
    args = parser.parse_args()
    args.normalize = not args.nonorm

    global N_EPOCHS, EPOCH_LENGTH, EVAL_SAMPLES, MIN_POOL_SIZE
    if args.sweep:
        N_EPOCHS = SWEEP_N_EPOCHS
        MIN_POOL_SIZE = SWEEP_MIN_POOL_SIZE
        EPOCH_LENGTH = SWEEP_EPOCH_LENGTH
        EVAL_SAMPLES = SWEEP_EVAL_SAMPLES
    if args.fast:
        N_EPOCHS = FAST_N_EPOCHS
        EPOCH_LENGTH = FAST_EPOCH_LENGTH
        EVAL_SAMPLES = FAST_EVAL_SAMPLES
        MIN_POOL_SIZE = FAST_MIN_POOL_SIZE

    else:
        if args.render:
            print("WARNING: Algorithm will be slow because render is on.")

    algo_settings = get_algo_settings_from_args(args)
    env_params = get_env_params_from_args(args)
    if args.benchmark:
        benchmark(args)
    elif args.sweep:
        sweep(args.name, env_params, algo_settings)
    else:
        algorithm_launcher = algo_settings['algorithm_launcher']
        algo_params = algo_settings['algo_params']
        print("algo_params =")
        print(algo_params)
        env_params = get_env_params_from_args(args)
        for i in range(args.num_seeds):
            algorithm_launcher(algo_params, env_params, args.name,
                               seed=args.seed+i)


if __name__ == "__main__":
    main()
