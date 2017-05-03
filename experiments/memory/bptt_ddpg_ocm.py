"""
Use an oracle qfunction to train a policy in bptt-ddpg style.
"""
import copy
import joblib
from hyperopt import hp
import numpy as np
import tensorflow as tf
import random

from railrl.envs.memory.one_char_memory import (
    OneCharMemoryOutputRewardMag,
    OneCharMemoryEndOnly,
)
from railrl.exploration_strategies.action_aware_memory_strategy import \
    ActionAwareMemoryStrategy
from railrl.launchers.launcher_util import (
    run_experiment,
)
from railrl.misc.hyperparameter import DeterministicHyperparameterSweeper
from railrl.policies.memory.action_aware_memory_policy import \
    ActionAwareMemoryPolicy
from railrl.policies.memory.lstm_memory_policy import (
    OutputAwareLstmCell,
    LstmLinearCell,
    FrozenHiddenLstmLinearCell,
    IRnnCell,
    LinearRnnCell,
)
from railrl.algos.writeback_bptt_ddpt import WritebackBpttDDPG
from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.algos.sum_bptt_ddpg import SumBpttDDPG
from railrl.exploration_strategies.noop import NoopStrategy
from railrl.exploration_strategies.onehot_sampler import OneHotSampler
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy

from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.exploration_strategies.action_aware_memory_strategy import \
    ActionAwareMemoryStrategy
from railrl.launchers.launcher_util import (
    run_experiment_here,
    create_base_log_dir,
)
from railrl.misc.hypopt import optimize_and_save


def run_ocm_experiment(variant):
    from railrl.algos.oracle_bptt_ddpg import (
        OracleBpttDDPG,
        OracleUnrollBpttDDPG,
    )
    from railrl.algos.regress_q_bptt_ddpg import RegressQBpttDdpg
    from railrl.qfunctions.memory.oracle_qfunction import OracleQFunction
    from railrl.qfunctions.memory.oracle_unroll_qfunction import (
        OracleUnrollQFunction
    )
    from railrl.exploration_strategies.product_strategy import ProductStrategy
    from railrl.envs.memory.continuous_memory_augmented import (
        ContinuousMemoryAugmented
    )
    from railrl.policies.memory.lstm_memory_policy import LstmMemoryPolicy
    from railrl.launchers.launcher_util import (
        set_seed,
    )
    from railrl.data_management.ocm_subtraj_replay_buffer import (
        OcmSubtrajReplayBuffer
    )
    from railrl.qfunctions.memory.hint_mlp_memory_qfunction import (
        HintMlpMemoryQFunction
    )
    from railrl.qfunctions.memory.mlp_memory_qfunction import MlpMemoryQFunction
    from os.path import exists

    """
    Set up experiment variants.
    """
    seed = variant['seed']
    load_policy_file = variant.get('load_policy_file', None)
    memory_dim = variant['memory_dim']
    oracle_mode = variant['oracle_mode']
    env_class = variant['env_class']
    env_params = variant['env_params']
    ddpg_params = variant['ddpg_params']
    policy_params = variant['policy_params']
    qf_params = variant['qf_params']
    es_params = variant['es_params']

    env_es_class = es_params['env_es_class']
    env_es_params = es_params['env_es_params']
    memory_es_class = es_params['memory_es_class']
    memory_es_params = es_params['memory_es_params']
    noise_action_to_memory = es_params['noise_action_to_memory']
    num_bptt_unrolls = ddpg_params['num_bptt_unrolls']
    set_seed(seed)

    """
    Code for running the experiment.
    """

    ocm_env = env_class(**env_params)
    env_action_dim = ocm_env.action_space.flat_dim
    env_obs_dim = ocm_env.observation_space.flat_dim
    H = ocm_env.horizon
    env = ContinuousMemoryAugmented(
        ocm_env,
        num_memory_states=memory_dim,
    )

    policy = None
    qf = None
    if load_policy_file is not None and exists(load_policy_file):
        with tf.Session():
            data = joblib.load(load_policy_file)
            policy = data['policy']
            qf = data['qf']
    env_strategy = env_es_class(
        env_spec=ocm_env.spec,
        **env_es_params,
    )
    write_strategy = memory_es_class(
        env_spec=env.memory_spec,
        **memory_es_params
    )
    if noise_action_to_memory:
        es = ActionAwareMemoryStrategy(
            env_strategy=env_strategy,
            write_strategy=write_strategy,
        )
        policy = policy or ActionAwareMemoryPolicy(
            name_or_scope="noisy_policy",
            action_dim=env_action_dim,
            memory_dim=memory_dim,
            env_spec=env.spec,
            **policy_params
        )
    else:
        es = ProductStrategy([env_strategy, write_strategy])
        policy = policy or LstmMemoryPolicy(
            name_or_scope="policy",
            action_dim=env_action_dim,
            memory_dim=memory_dim,
            env_spec=env.spec,
            num_env_obs_dims_to_use=env_params['n'] + 1,
            **policy_params
        )

    ddpg_params = ddpg_params.copy()
    unroll_through_target_policy = ddpg_params.pop(
        'unroll_through_target_policy',
        False,
    )
    if oracle_mode == 'none':
        qf = MlpMemoryQFunction(
            name_or_scope="critic",
            env_spec=env.spec,
        )
        algo_class = variant['algo_class']
    elif oracle_mode == 'unroll':
        qf = OracleUnrollQFunction(
            name_or_scope="oracle_unroll_critic",
            env=env,
            policy=policy,
            num_bptt_unrolls=num_bptt_unrolls,
            env_obs_dim=env_obs_dim,
            env_action_dim=env_action_dim,
            max_horizon_length=H,
            env_spec=env.spec,
        )
        algo_class = OracleUnrollBpttDDPG
        ddpg_params['unroll_through_target_policy'] = (
            unroll_through_target_policy
        )
    elif oracle_mode == 'hint':
        qf = HintMlpMemoryQFunction(
            name_or_scope="hint_critic",
            hint_dim=env_action_dim,
            env_spec=env.spec,
            max_time=H,
        )
        algo_class = OracleBpttDDPG
    elif oracle_mode == 'oracle':
        qf = OracleQFunction(
            name_or_scope="oracle_critic",
            env=env,
            env_spec=env.spec,
        )
        algo_class = OracleBpttDDPG
    elif oracle_mode == 'regress':
        regress_params = variant['regress_params']
        if regress_params.pop('use_hint_qf', False):
            qf = qf or HintMlpMemoryQFunction(
                name_or_scope="hint_critic",
                hint_dim=env_action_dim,
                max_time=H,
                env_spec=env.spec,
                **qf_params
            )
        else:
            qf = qf or MlpMemoryQFunction(
                name_or_scope="critic",
                env_spec=env.spec,
                **qf_params
            )
        oracle_qf = OracleUnrollQFunction(
            name_or_scope="oracle_unroll_critic",
            env=env,
            policy=policy,
            num_bptt_unrolls=num_bptt_unrolls,
            env_obs_dim=env_obs_dim,
            env_action_dim=env_action_dim,
            max_horizon_length=H,
            env_spec=env.spec,
        )
        algo_class = RegressQBpttDdpg
        ddpg_params['oracle_qf'] = oracle_qf
        ddpg_params.update(regress_params)
    else:
        raise Exception("Unknown mode: {}".format(oracle_mode))

    algorithm = algo_class(
        env,
        es,
        policy,
        qf,
        env_obs_dim=env_obs_dim,
        env_action_dim=env_action_dim,
        replay_buffer_class=OcmSubtrajReplayBuffer,
        **ddpg_params
    )

    algorithm.train()
    return algorithm


def get_ocm_score(variant):
    algorithm = run_ocm_experiment(variant)
    scores = algorithm.epoch_scores
    return np.mean(scores[-3:])


def create_run_experiment_multiple_seeds(n_seeds):
    def run_experiment_with_multiple_seeds(variant):
        scores = []
        for i in range(n_seeds):
            variant['seed'] = str(int(variant['seed']) + i)
            exp_prefix = variant['exp_prefix']
            scores.append(run_experiment_here(
                get_ocm_score,
                exp_prefix=exp_prefix,
                variant=variant,
                exp_id=i,
            ))
        return np.mean(scores)

    return run_experiment_with_multiple_seeds


if __name__ == '__main__':
    n_seeds = 1
    mode = 'here'
    exp_prefix = "dev-bptt-ddpg-ocm"
    run_mode = 'none'
    version = 'dev'
    # n_seeds = 5
    # mode = 'ec2'
    # exp_prefix = '5-1-no=bptt'
    # run_mode = 'none'
    # version = 'bptt-ddpg-bellman-stochastic-rnn'

    """
    DDPG Params
    """
    n_batches_per_epoch = 100
    n_batches_per_eval = 64
    batch_size = 32
    n_epochs = 20
    memory_dim = 2
    min_pool_size = max(n_batches_per_epoch, batch_size)
    replay_pool_size = 100000
    bpt_bellman_error_weight = 1.

    """
    Algorithm Selection
    """
    oracle_mode = 'regress'
    algo_class = BpttDDPG
    unroll_through_target_policy = False

    """
    Policy Params
    """
    # policy_rnn_cell_class = LinearRnnCell
    # policy_rnn_cell_class = OutputAwareLstmCell
    # policy_rnn_cell_class = IRnnCell
    policy_rnn_cell_class = LstmLinearCell
    # policy_rnn_cell_class = FrozenHiddenLstmLinearCell
    load_policy_file = (
        '/home/vitchyr/git/rllab-rail/railrl/data/reference/expert'
        '/ocm_reward_magnitude5_H6_nbptt6_100p'
        '/params.pkl'
    )
    load_policy_file = 'none'

    """
    Env param
    """
    # env_class = OneCharMemoryOutputRewardMag
    env_class = OneCharMemoryEndOnly
    H = 6
    num_values = 2
    zero_observation = True
    env_output_target_number = False
    env_output_time = False

    """
    Algo params
    """
    num_extra_qf_updates = 100
    qf_learning_rate = 1e-3
    policy_learning_rate = 1e-3
    soft_target_tau = 0.01
    qf_weight_decay = 0.01
    num_bptt_unrolls = min(6, H)
    qf_total_loss_tolerance = -9999
    max_num_q_updates = 100
    train_policy = True
    extra_qf_training_mode = 'none'
    freeze_hidden = False

    """
    Regression Params
    """
    env_grad_distance_weight = 0.
    write_grad_distance_weight = 0.
    qf_grad_mse_from_one_weight = 0.
    regress_onto_values_weight = 0.
    bellman_error_weight = 1.
    use_hint_qf = False
    use_time = False

    """
    Exploration params
    """
    env_es_class = NoopStrategy
    # env_es_class = OneHotSampler
    # env_es_class = OUStrategy
    env_es_params = dict(
        max_sigma=1.0,
        min_sigma=0.5,
        decay_period=500,
        softmax=True,
        laplace_weight=0.,
    )
    memory_es_class = NoopStrategy
    # memory_es_class = OneHotSampler
    # memory_es_class = OUStrategy
    memory_es_params = dict(
        max_sigma=0.5,
        min_sigma=0.1,
        decay_period=1000,
        softmax=True,
    )
    noise_action_to_memory = False

    """
    LSTM Cell params
    """
    use_peepholes = True
    env_noise_std = 0.
    memory_noise_std = 1.

    """
    Create them dict's
    """
    es_params = dict(
        env_es_class=env_es_class,
        env_es_params=env_es_params,
        memory_es_class=memory_es_class,
        memory_es_params=memory_es_params,
        noise_action_to_memory=noise_action_to_memory,
    )
    epoch_length = H * n_batches_per_epoch
    eval_samples = H * n_batches_per_eval
    max_path_length = H + 2
    ddpg_params = dict(
        batch_size=batch_size,
        n_epochs=n_epochs,
        min_pool_size=min_pool_size,
        replay_pool_size=replay_pool_size,
        epoch_length=epoch_length,
        eval_samples=eval_samples,
        max_path_length=max_path_length,
        num_bptt_unrolls=num_bptt_unrolls,
        unroll_through_target_policy=unroll_through_target_policy,
        freeze_hidden=freeze_hidden,
        qf_learning_rate=qf_learning_rate,
        policy_learning_rate=policy_learning_rate,
        discount=1.0,
        soft_target_tau=soft_target_tau,
        qf_weight_decay=qf_weight_decay,
        bpt_bellman_error_weight=bpt_bellman_error_weight,
        save_tf_graph=False,
    )
    regress_params = dict(
        use_hint_qf=use_hint_qf,
        qf_total_loss_tolerance=qf_total_loss_tolerance,
        max_num_q_updates=max_num_q_updates,
        train_policy=train_policy,
        env_grad_distance_weight=env_grad_distance_weight,
        write_grad_distance_weight=write_grad_distance_weight,
        qf_grad_mse_from_one_weight=qf_grad_mse_from_one_weight,
        regress_onto_values_weight=regress_onto_values_weight,
        bellman_error_weight=bellman_error_weight,
        num_extra_qf_updates=num_extra_qf_updates,
        extra_qf_training_mode=extra_qf_training_mode,
    )
    policy_params = dict(
        rnn_cell_class=policy_rnn_cell_class,
        rnn_cell_params=dict(
            use_peepholes=use_peepholes,
            env_noise_std=env_noise_std,
            memory_noise_std=memory_noise_std,
        )
    )
    qf_params = dict(
        # hidden_nonlinearity=tf.nn.relu,
        # output_nonlinearity=tf.nn.tanh,
        # hidden_nonlinearity=tf.identity,
        # output_nonlinearity=tf.identity,
        # embedded_hidden_sizes=[100, 100],
        # observation_hidden_sizes=[100, 100],
    )
    if use_hint_qf:
        qf_params['use_time'] = use_time
    # TODO(vitchyr): Oracle needs to use the true reward
    env_params = dict(
        n=num_values,
        num_steps=H,
        zero_observation=zero_observation,
        max_reward_magnitude=1,
        output_target_number=env_output_target_number,
        output_time=env_output_time,
    )
    variant = dict(
        exp_prefix=exp_prefix,
        memory_dim=memory_dim,
        algo_class=algo_class,
        version=version,
        load_policy_file=load_policy_file,
        oracle_mode=oracle_mode,
        env_class=env_class,
        env_params=env_params,
        ddpg_params=ddpg_params,
        policy_params=policy_params,
        qf_params=qf_params,
        regress_params=regress_params,
        es_params=es_params,
    )

    if run_mode == 'hyperopt':
        mem_gaussian_es_space = {
            'es_params.memory_es_params.max_sigma': hp.uniform(
                'es_params.memory_es_params.max_sigma',
                3.0,
                0.5,
            ),
            'es_params.memory_es_params.min_sigma': hp.uniform(
                'es_params.memory_es_params.min_sigma',
                0.2,
                0.0,
            ),
            'es_params.memory_es_params.decay_period': hp.qloguniform(
                'es_params.memory_es_params.decay_period',
                np.log(100),
                np.log(100000),
                1,
            )
        }
        mem_ou_es_space = copy.deepcopy(mem_gaussian_es_space)
        mem_ou_es_space['es_params.memory_es_params.theta'] = hp.uniform(
            'es_params.memory_es_params.theta',
            1.0,
            0.,
        ),
        mem_noop_strategy_space = {'es_params.memory_es_params': {}}
        search_space = {
            'es_params.memory_es_class': hp.choice(
                'es_params.memory_es_class',
                [
                    (GaussianStrategy, mem_ou_es_space),
                    (OUStrategy, mem_ou_es_space),
                    (NoopStrategy, mem_noop_strategy_space),
                ]
            ),
            'es_params.noise_action_to_memory': hp.choice(
                'es_params.noise_action_to_memory',
                [True, False],
            ),
            'seed': hp.randint('seed', 10000),
        }
        variant['es_params'].pop('memory_es_class')
        variant['es_params'].pop('memory_es_params')
        variant['es_params'].pop('noise_action_to_memory')

        base_log_dir = create_base_log_dir(exp_prefix=exp_prefix)

        optimize_and_save(
            base_log_dir,
            create_run_experiment_multiple_seeds(n_seeds=n_seeds),
            search_space=search_space,
            extra_function_kwargs=variant,
            maximize=True,
            verbose=True,
            load_trials=True,
            num_rounds=500,
            num_evals_per_round=1,
        )
    elif run_mode == 'grid':
        search_space = {
            'policy_params.rnn_cell_params.env_noise_std':
                [0., 0.1, 0.2, 0.4],
            'policy_params.rnn_cell_params.memory_noise_std':
                [0., 0.3, 1, 2],
            'ddpg_params.bpt_bellman_error_weight': [0., 0.5, 1, 5]
        }
        sweeper = DeterministicHyperparameterSweeper(search_space,
                                                     default_parameters=variant)
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            for i in range(n_seeds):
                run_experiment(
                    get_ocm_score,
                    exp_prefix=exp_prefix,
                    seed=i,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                )
    elif run_mode == 'custom_grid':
        for exp_id, (
            version,
            bpt_bellman_error_weight,
            memory_noise_std,
            env_noise_std
        ) in enumerate([
            ("Basic", 1., 1., 0.),
            ("No Bpt Bellman Error", 0., 1., 0.),
            ("No Stochastic RNN", 1., 0., 1.),
            ("No Bpt Bellman Error, No Stochastic RNN", 0., 0., 1.),
        ]):
            exp_id += 1000
            variant['version'] = version
            variant['policy_params']['rnn_cell_params']['env_noise_std'] = (
                env_noise_std
            )
            variant['policy_params']['rnn_cell_params']['memory_noise_std'] = (
                memory_noise_std
            )
            variant['ddpg_params']['bpt_bellman_error_weight'] = (
                bpt_bellman_error_weight
            )
            for seed in range(n_seeds):
                run_experiment(
                    get_ocm_score,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                )
    else:
        for _ in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                run_ocm_experiment,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=0,
            )
