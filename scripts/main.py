"""Test different rl algorithms."""
import argparse
import copy

import tensorflow as tf

from rlkit.launchers.rnn_launchers import (
    bptt_launcher,
)
from rlkit.launchers.algo_launchers import (
    mem_ddpg_launcher,
    my_ddpg_launcher,
    naf_launcher,
    random_action_launcher,
    get_env_settings)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.misc import hyperparameter as hp

BATCH_SIZE = 32
N_EPOCHS = 100
EPOCH_LENGTH = 3 * 32 * 10
EVAL_SAMPLES = 3 * 64
DISCOUNT = 0.99
QF_LEARNING_RATE = 1e-3
POLICY_LEARNING_RATE = 1e-4
BATCH_LEARNING_RATE = 1e-2
SOFT_TARGET_TAU = 1e-2
REPLAY_POOL_SIZE = 1000000
MIN_POOL_SIZE = 10000
SCALE_REWARD = 1.0
QF_WEIGHT_DECAY = 0.0001
MAX_PATH_LENGTH = 1000
N_UPDATES_PER_TIME_STEP = 5
BATCH_NORM_PARAMS = None  # None = off, {} = default params

# Sweep settings
SWEEP_N_EPOCHS = 20
SWEEP_EPOCH_LENGTH = 10000
SWEEP_EVAL_SAMPLES = 10000
SWEEP_MIN_POOL_SIZE = 10000

# Fast settings
FAST_N_EPOCHS = 5
FAST_EPOCH_LENGTH = 5
FAST_EVAL_SAMPLES = 5
FAST_MIN_POOL_SIZE = 5
FAST_MAX_PATH_LENGTH = 5

NUM_SEEDS_PER_CONFIG = 3
NUM_HYPERPARAMETER_CONFIGS = 50

# One character memory settings
OCM_N = 4
OCM_NUM_STEPS = 10
OCM_REWARD_FOR_REMEMBERING = 1
OCM_MAX_REWARD_MAGNITUDE = 1


def get_launch_settings_list_from_args(args):
    render = args.render

    def get_launch_settings(algo_name):
        """
        Return a dictionary of the form
        {
            'algo_params': algo_params to pass to run_algorithm
            'variant': variant to pass to run_algorithm
        }
        :param algo_name: Name of the algorithm to run.
        :return:
        """
        sweeper = hp.RandomHyperparameterSweeper()
        algo_params = {}
        if algo_name == 'ddpg' or algo_name == 'mddpg':
            sweeper = hp.RandomHyperparameterSweeper([
                hp.LogFloatParam("qf_learning_rate", 1e-5, 1e-2),
                hp.LogFloatParam("policy_learning_rate", 1e-6, 1e-3),
                hp.LogFloatParam("reward_scale", 10.0, 0.001),
                hp.LogFloatParam("soft_target_tau", 1e-5, 1e-2),
            ])
            algo_params = get_ddpg_params()
            algo_params['render'] = render
            variant = {
                'qf_params': dict(
                    embedded_hidden_sizes=(100,),
                    observation_hidden_sizes=(100,),
                    hidden_nonlinearity=tf.nn.relu,
                ),
                'policy_params': dict(
                    observation_hidden_sizes=(100, 100),
                    hidden_nonlinearity=tf.nn.relu,
                )
            }
            if algo_name == 'ddpg':
                algorithm_launcher = my_ddpg_launcher
                variant['Algorithm'] = 'DDPG'
                variant['policy_params']['output_nonlinearity'] = tf.nn.tanh
            else:
                algorithm_launcher = mem_ddpg_launcher
                variant['Algorithm'] = 'Memory-DDPG'
        elif algo_name == 'naf':
            sweeper = hp.RandomHyperparameterSweeper([
                hp.LogFloatParam("qf_learning_rate", 1e-5, 1e-2),
                hp.LogFloatParam("reward_scale", 10.0, 0.001),
                hp.LogFloatParam("soft_target_tau", 1e-6, 1e-1),
                hp.LogFloatParam("qf_weight_decay", 1e-7, 1e-1),
            ])
            algo_params = get_my_naf_params()
            algo_params['render'] = render
            algorithm_launcher = naf_launcher
            variant = {
                'Algorithm': 'NAF',
                'exploration_strategy_params': {
                    'sigma': 0.15
                },
            }
        elif algo_name == 'random':
            algorithm_launcher = random_action_launcher
            variant = {'Algorithm': 'Random'}
        elif algo_name == 'bptt':
            algorithm_launcher = bptt_launcher
            variant = {'Algorithm': 'BPTT'}
        else:
            raise Exception("Algo name not recognized: " + algo_name)

        # bn_sweeper = hp.RandomHyperparameterSweeper([
        #     hp.EnumParam("decay", [0.9, 0.99, 0.999, 0.9999]),
        #     hp.LogFloatParam("epsilon", 1e-3, 1e-7),
        #     hp.EnumParam("enable_offset", [True, False]),
        #     hp.EnumParam("enable_scale", [True, False]),
        # ])
        bn_sweeper = None
        return {
            'sweeper': sweeper,
            'batch_norm_sweeper': bn_sweeper,
            'variant': variant,
            'algo_params': algo_params,
            'algorithm_launcher': algorithm_launcher,
            'batch_norm_params': BATCH_NORM_PARAMS
        }

    return [get_launch_settings(algo_name) for algo_name in args.algo]


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


def run_algorithm(
        launch_settings,
        env_params,
        exp_prefix,
        seed,
        exp_id=1,
        **kwargs):
    """
    Launch an algorithm
    :param launch_settings: See get_launch_settings_list_from_args
    :param env_params: See get_env_settings
    :param exp_prefix: Experiment prefix
    :param seed: Experiment seed
    :param exp_id: Experiment ID # to identify it later (e.g. for plotting data)
    :param kwargs: Other kwargs to pass to run_experiment_lite
    :return:
    """
    variant = launch_settings['variant']
    variant['env_params'] = env_params
    variant['algo_params'] = launch_settings['algo_params']
    variant['batch_norm_params'] = launch_settings['batch_norm_params']
    variant['exp_id'] = exp_id

    env_settings = get_env_settings(**env_params)
    variant['Environment'] = env_settings['name']
    algorithm_launcher = launch_settings['algorithm_launcher']

    run_experiment(
        algorithm_launcher,
        exp_prefix,
        seed,
        variant,
        **kwargs)


def sweep(exp_prefix, env_params, launch_settings_, **kwargs):
    launch_settings = copy.deepcopy(launch_settings_)
    sweeper = launch_settings['sweeper']
    bn_sweeper = launch_settings['batch_norm_sweeper']
    default_params = launch_settings['algo_params']
    default_bn_params = launch_settings['batch_norm_params']
    exp_id = 0
    # So far, only support bn sweeper in random mode
    assert bn_sweeper is None or (
        not isinstance(sweeper, hp.DeterministicHyperparameterSweeper) and
        not isinstance(bn_sweeper, hp.DeterministicHyperparameterSweeper)
    )
    if isinstance(sweeper, hp.DeterministicHyperparameterSweeper):
        for params_dict in sweeper.iterate_hyperparameters():
            exp_id += 1
            algo_params = dict(default_params, **params_dict)
            for seed in range(NUM_SEEDS_PER_CONFIG):
                launch_settings['algo_params'] = algo_params
                run_algorithm(launch_settings, env_params, exp_prefix, seed,
                              exp_id=exp_id,
                              **kwargs)
    else:
        for i in range(NUM_HYPERPARAMETER_CONFIGS):
            exp_id += 1
            algo_params = dict(default_params,
                               **sweeper.generate_random_hyperparameters())
            if bn_sweeper is None:
                bn_params = default_bn_params
            else:
                bn_params = dict(default_bn_params,
                                 **bn_sweeper.generate_random_hyperparameters())
            for seed in range(NUM_SEEDS_PER_CONFIG):
                launch_settings['algo_params'] = algo_params
                launch_settings['batch_norm_params'] = bn_params
                run_algorithm(launch_settings, env_params, exp_prefix, seed,
                              exp_id=exp_id,
                              **kwargs)


def get_env_params_list_from_args(args):
    envs_params_list = []
    num_memory_states = args.num_memory_states
    if 'gym' in args.env:
        envs_params_list = [
            dict(
                env_id='gym',
                normalize_env=args.normalize,
                gym_name=gym_name,
                num_memory_states=num_memory_states,
            )
            for gym_name in args.gym
            ]

    init_env_params = {}
    init_env_params['n'] = OCM_N
    init_env_params['num_steps'] = OCM_NUM_STEPS
    init_env_params['reward_for_remembering'] = OCM_REWARD_FOR_REMEMBERING
    init_env_params['max_reward_magnitude'] = OCM_MAX_REWARD_MAGNITUDE
    if args.ocm_horizon:
        init_env_params['num_steps'] = args.ocm_horizon
    return envs_params_list + [dict(
        env_id=env,
        normalize_env=args.normalize,
        gym_name="",
        num_memory_states=num_memory_states,
        init_env_params=init_env_params,
    ) for env in args.env if env != 'gym']


def main():
    env_choices = ['ocm', 'ocme', 'point', 'cheetah']
    algo_choices = ['mddpg', 'ddpg', 'naf', 'bptt', 'random']
    mode_choices = ['local', 'local_docker', 'ec2']
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action='store_true',
                        help="Sweep _hyperparameters for my DDPG.")
    parser.add_argument("--render", action='store_true',
                        help="Render the environment.")
    parser.add_argument("--env",
                        default=['ocme'],
                        help="Environment to test. If env is 'gym' then you "
                             "must pass in argument to the '--gym' option.",
                        nargs='+',
                        choices=env_choices)
    parser.add_argument("--gym",
                        nargs='+',
                        help="Gym environment name (e.g. Cartpole-V1) to test. "
                             "Must pass 'gym' to the '--env' option to use "
                             "this.")
    parser.add_argument("--name", default='default-icml2017',
                        help='Experiment prefix')
    parser.add_argument("--fast", action='store_true',
                        help=('Run a quick experiment. Intended for debugging. '
                              'Overrides sweep settings'))
    parser.add_argument("--normalize", action='store_true',
                        help="Normalize the environment")
    parser.add_argument("--algo",
                        default=['bptt'],
                        help='Algorithm to run.',
                        nargs='+',
                        choices=algo_choices)
    parser.add_argument("--seed", default=0,
                        type=int,
                        help='Seed')
    parser.add_argument("--num_seeds", default=NUM_SEEDS_PER_CONFIG, type=int,
                        help="Run this many seeds, starting with --seed.")
    parser.add_argument("--mode",
                        default='local',
                        help="Mode to run experiment.",
                        choices=mode_choices,
                        )
    parser.add_argument("--notime", action='store_true',
                        help="Disable time prefix to python command.")
    parser.add_argument("--profile", action='store_true',
                        help="Use cProfile to time the python script.")
    parser.add_argument("--profile_file",
                        help="Where to save .prof file output of cProfiler. "
                             "If set, --profile is forced to be true.")
    parser.add_argument("--num_memory_states", default=0,
                        type=int,
                        help='Number of memory states. If positive, '
                             'the environment is wrapped in a '
                             'ContinuousMemoryAugmented env')
    parser.add_argument("--ocm_horizon", default=100,
                        type=int,
                        help='For how long the character must be memorized.')
    args = parser.parse_args()
    args.time = not args.notime

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

    kwargs = dict(
        time=not args.notime,
        save_profile=args.profile or args.profile_file is not None,
        mode=args.mode
    )
    if args.profile_file:
        kwargs['profile_file'] = args.profile_file
    for env_params in get_env_params_list_from_args(args):
        for launcher_settings in get_launch_settings_list_from_args(args):
            if args.sweep:
                sweep(
                    args.name,
                    env_params,
                    launcher_settings,
                    **kwargs
                )
            else:
                for i in range(args.num_seeds):
                    run_algorithm(
                        launcher_settings,
                        env_params,
                        args.name,
                        args.seed + i,
                        exp_id=i,
                        **kwargs
                    )


if __name__ == "__main__":
    main()
