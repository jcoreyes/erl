"""Test different rl algorithms."""
import argparse

from algo_launchers import (
    test_my_ddpg,
    test_my_naf,
    test_random_ddpg,
    test_shane_ddpg,
)
from misc import hyperparameter as hp
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub

BATCH_SIZE = 64
N_EPOCHS = 1000
EPOCH_LENGTH = 10000
EVAL_SAMPLES = 10000
DISCOUNT = 0.99
CRITIC_LEARNING_RATE = 1e-3
ACTOR_LEARNING_RATE = 1e-4
SOFT_TARGET_TAU = 0.01
REPLAY_POOL_SIZE = 1000000
MIN_POOL_SIZE = 1000
SCALE_REWARD = 1.0
Q_WEIGHT_DECAY = 0.0
MAX_PATH_LENGTH = 1000

# Sweep settings
SWEEP_N_EPOCHS = 50
SWEEP_MIN_POOL_SIZE = BATCH_SIZE

# Fast settings
FAST_N_EPOCHS = 10
FAST_EPOCH_LENGTH = 50
FAST_EVAL_SAMPLES = 10
FAST_MIN_POOL_SIZE = 2

NUM_SEEDS_PER_CONFIG = 2
NUM_HYPERPARAMETER_CONFIGS = 50


def get_env_settings(args):
    env_name = args.env
    if env_name == 'cart':
        env = CartpoleEnv()
        name = "Cartpole"
    elif env_name == 'cheetah':
        env = HalfCheetahEnv()
        name = "HalfCheetah"
    elif env_name == 'point':
        env = normalize(GymEnv("Pointmass-v1", record_video=False,
                               log_dir='/tmp/gym-test',  # Ignore gym log.
                               record_log=False))
        name = "Pointmass"
    else:
        raise Exception("Unknown env: {0}".format(env_name))

    return dict(
        env=env,
        name=name,
    )


def get_algo_settings(args):
    algo_name = args.algo
    if algo_name == 'ddpg':
        sweeper = hp.HyperparameterSweeper([
            hp.LogFloatParam("soft_target_tau", 0.005, 0.1),
            hp.LogFloatParam("scale_reward", 10.0, 0.01),
            hp.LogFloatParam("Q_weight_decay", 1e-7, 1e-1),
        ])
        params = get_my_ddpg_params()
        test_function = test_my_ddpg
    elif algo_name == 'shane-ddpg':
        sweeper = hp.HyperparameterSweeper([
            hp.LogFloatParam("soft_target_tau", 0.005, 0.1),
            hp.LogFloatParam("scale_reward", 10.0, 0.01),
            hp.LogFloatParam("Q_weight_decay", 1e-7, 1e-1),
        ])
        params = get_ddpg_params()
        test_function = test_shane_ddpg
    elif algo_name == 'ddpg-quad-qf':
        sweeper = hp.HyperparameterSweeper([
            hp.LogFloatParam("soft_target_tau", 0.005, 0.1),
            hp.LogFloatParam("scale_reward", 10.0, 0.01),
            hp.LogFloatParam("Q_weight_decay", 1e-7, 1e-1),
        ])
        params = get_my_ddpg_params()
        test_function = test_quad_critic_ddpg
    elif algo_name == 'naf':
        sweeper = hp.HyperparameterSweeper([
            hp.LogFloatParam("qf_learning_rate", 1e-4, 1e-2),
            hp.LogFloatParam("scale_reward", 10.0, 0.01),
            hp.LogFloatParam("soft_target_tau", 0.005, 0.1),
            hp.LinearIntParam("n_updates_per_time_step", 1, 5),
        ])
        params = get_my_naf_params()
        test_function = test_my_naf
    elif algo_name == 'random':
        sweeper = hp.HyperparameterSweeper()
        params = {}
        test_function = test_random_ddpg

    else:
        raise Exception("Algo name not recognized: " + algo_name)

    params['render'] = args.render
    return {
        'sweeper': sweeper,
        'algo_params': params,
        'test_function': test_function,
    }


def get_ddpg_params():
    return dict(
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        epoch_length=EPOCH_LENGTH,
        eval_samples=EVAL_SAMPLES,
        discount=DISCOUNT,
        policy_learning_rate=ACTOR_LEARNING_RATE,
        qf_learning_rate=CRITIC_LEARNING_RATE,
        soft_target_tau=SOFT_TARGET_TAU,
        replay_pool_size=REPLAY_POOL_SIZE,
        min_pool_size=MIN_POOL_SIZE,
        scale_reward=SCALE_REWARD,
        max_path_length=MAX_PATH_LENGTH,
    )


def get_my_ddpg_params():
    params = get_ddpg_params()
    params['Q_weight_decay'] = Q_WEIGHT_DECAY
    return params


def get_my_naf_params():
    return dict(
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        epoch_length=EPOCH_LENGTH,
        eval_samples=EVAL_SAMPLES,
        discount=DISCOUNT,
        qf_learning_rate=CRITIC_LEARNING_RATE,
        soft_target_tau=SOFT_TARGET_TAU,
        replay_pool_size=REPLAY_POOL_SIZE,
        min_pool_size=MIN_POOL_SIZE,
        scale_reward=SCALE_REWARD,
        max_path_length=MAX_PATH_LENGTH,
        Q_weight_decay=Q_WEIGHT_DECAY,
        n_updates_per_time_step=5,
    )


def sweep(exp_prefix, env_settings, algo_settings):
    sweeper = algo_settings['sweeper']
    test_function = algo_settings['test_function']
    default_params = algo_settings['algo_params']
    env = env_settings['env']
    env_name = env_settings['name']
    for i in range(NUM_HYPERPARAMETER_CONFIGS):
        for seed in range(NUM_SEEDS_PER_CONFIG):
            params = dict(default_params,
                          **sweeper.generate_random_hyperparameters())
            test_function(env, exp_prefix, env_name, seed=seed + 1,
                          **params)


def main():
    env_choices = ['cheetah', 'cart', 'point']
    algo_choices = ['ddpg', 'naf', 'shane-ddpg', 'random', 'ddpg-quad-qf']
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action='store_true',
                        help="Run benchmarks.")
    parser.add_argument("--sweep", action='store_true',
                        help="Sweep hyperparameters for my DDPG.")
    parser.add_argument("--render", action='store_true',
                        help="Render the environment.")
    parser.add_argument("--env", default='cart',
                        help="Test algo on 'cart' or 'cheetah'.",
                        choices=env_choices)
    parser.add_argument("--name", default='default',
                        help='Experiment prefix')
    parser.add_argument("--fast", action='store_true',
                        help='Run a quick experiment. Intended for debugging. Trumps sweep settings')
    parser.add_argument("--algo", default='ddpg',
                        help='Algo',
                        choices=algo_choices)
    parser.add_argument("--seed", default=1,
                        type=int,
                        help='Seed')
    args = parser.parse_args()

    if args.sweep:
        global N_EPOCHS, MIN_POOL_SIZE
        N_EPOCHS = SWEEP_N_EPOCHS
        MIN_POOL_SIZE = SWEEP_MIN_POOL_SIZE
    if args.fast:
        global N_EPOCHS, EPOCH_LENGTH, EVAL_SAMPLES, MIN_POOL_SIZE
        N_EPOCHS = FAST_N_EPOCHS
        EPOCH_LENGTH = FAST_EPOCH_LENGTH
        EVAL_SAMPLES = FAST_EVAL_SAMPLES
        MIN_POOL_SIZE = FAST_MIN_POOL_SIZE

    else:
        if args.render:
            print("WARNING: Algorithm will be slow because render is on.")

    stub(globals())

    algo_settings = get_algo_settings(args)
    env_settings = get_env_settings(args)
    if args.sweep:
        sweep(args.name, env_settings, algo_settings)
    else:
        test_function = algo_settings['test_function']
        algo_params = algo_settings['algo_params']
        env = env_settings['env']
        env_name = env_settings['name']
        test_function(env, args.name, env_name, seed=args.seed, **algo_params)


if __name__ == "__main__":
    main()
