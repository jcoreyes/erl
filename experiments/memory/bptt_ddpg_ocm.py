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
from railrl.misc.hyperparameter import (
    DeterministicHyperparameterSweeper,
    RandomHyperparameterSweeper,
    LogFloatParam,
    LogFloatOffsetParam,
    LinearFloatParam,
)
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
    from railrl.algos.oracle_bptt_ddpg import OracleBpttDdpg
    from railrl.algos.meta_bptt_ddpg import MetaBpttDdpg
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
    elif oracle_mode == 'oracle' or oracle_mode == 'meta':
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
        algo_class = OracleBpttDdpg
        ddpg_params['oracle_qf'] = oracle_qf
        ddpg_params.update(regress_params)
    else:
        raise Exception("Unknown mode: {}".format(oracle_mode))
    if oracle_mode == 'meta':
        meta_qf = MlpMemoryQFunction(
            name_or_scope="meta_critic",
            env_spec=env.spec,
            **qf_params
        )
        algo_class = MetaBpttDdpg
        meta_params = variant['meta_params']
        ddpg_params['meta_qf'] = meta_qf
        ddpg_params.update(meta_params)

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
        print("VARIANT", variant)
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

    # n_seeds = 3
    # mode = 'ec2'
    # exp_prefix = '5-5-hyperopt-meta-no-flags'
    # run_mode = 'random'
    # version = 'dev'
    num_hp_settings = 100

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
    episode_boundary_flags = False

    """
    DDPG Params
    """
    n_batches_per_epoch = 100
    n_batches_per_eval = 64
    batch_size = 32
    n_epochs = 25
    memory_dim = 20
    min_pool_size = 320
    replay_pool_size = 100000
    bpt_bellman_error_weight = 2.043625554091334

    """
    Algorithm Selection
    """
    oracle_mode = 'meta'
    algo_class = BpttDDPG

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
    Algo params
    """
    num_extra_qf_updates = 10
    qf_learning_rate = 1e-3
    policy_learning_rate = 1e-3
    soft_target_tau = 0.01
    qf_weight_decay = 0.
    num_bptt_unrolls = 4
    qf_total_loss_tolerance = 0.1
    max_num_q_updates = 10000
    train_policy = True
    extra_qf_training_mode = 'none'
    freeze_hidden = False
    extra_train_period = 100

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
    use_target = False
    use_oracle_qf = False
    unroll_through_target_policy = False

    """
    Meta-critic Params
    """
    meta_qf_learning_rate = 0.0043686912042467125
    meta_qf_output_weight = 0.5895080878682102
    qf_output_weight = 1

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
    env_noise_std = 0.7834798765148419
    memory_noise_std = -1.3624080142760144

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
        extra_train_period=extra_train_period,
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
        use_oracle_qf=use_oracle_qf,
        unroll_through_target_policy=unroll_through_target_policy,
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
        qf_params['use_target'] = use_target
    meta_params = dict(
        meta_qf_learning_rate=meta_qf_learning_rate,
        meta_qf_output_weight=meta_qf_output_weight,
        qf_output_weight=qf_output_weight,
    )
    env_params = dict(
        n=num_values,
        num_steps=H,
        zero_observation=zero_observation,
        max_reward_magnitude=1,
        output_target_number=env_output_target_number,
        output_time=env_output_time,
        episode_boundary_flags=episode_boundary_flags,
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
        meta_params=meta_params,
    )

    if run_mode == 'hyperopt':
        search_space = {
            'policy_params.rnn_cell_params.env_noise_std': hp.uniform(
                'policy_params.rnn_cell_params.env_noise_std',
                np.log(0.001),
                np.log(5),
            ),
            'policy_params.rnn_cell_params.memory_noise_std': hp.uniform(
                'policy_params.rnn_cell_params.memory_noise_std',
                np.log(0.001),
                np.log(5),
            ),
            'ddpg_params.bpt_bellman_error_weight': hp.loguniform(
                'ddpg_params.bpt_bellman_error_weight',
                np.log(0.01),
                np.log(1000),
            ),
            'meta_params.meta_qf_learning_rate': hp.loguniform(
                'meta_params.meta_qf_learning_rate',
                np.log(1e-5),
                np.log(1e-2),
            ),
            'meta_params.meta_qf_output_weight': hp.loguniform(
                'meta_params.meta_qf_output_weight',
                np.log(1e-1),
                np.log(1000),
            ),
            'seed': hp.randint('seed', 10000),
        }

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
            # 'policy_params.rnn_cell_params.env_noise_std': [0., 1.],
            # 'policy_params.rnn_cell_params.memory_noise_std': [0., 1.],
            # 'ddpg_params.bpt_bellman_error_weight': [0, 1, 10, 100],
            'meta_params.meta_qf_learning_rate': [1e-3, 1e-4],
            'meta_params.meta_qf_output_weight': [0, 0.1, 1, 5, 10],
            # 'meta_params.qf_output_weight': [0, 1],
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
    elif run_mode == 'random':
        sweeper = RandomHyperparameterSweeper(
            hyperparameters=[
                LinearFloatParam(
                    'policy_params.rnn_cell_params.env_noise_std', 0, 1
                ),
                LinearFloatParam(
                    'policy_params.rnn_cell_params.memory_noise_std', 0, 1
                ),
                LogFloatOffsetParam(
                    'ddpg_params.bpt_bellman_error_weight', 1, 1001, -1
                ),
                LogFloatParam('meta_params.meta_qf_learning_rate', 1e-5, 1e-2),
                LogFloatOffsetParam(
                    'meta_params.meta_qf_output_weight', 1e-3, 1e3, -1e-3
                ),
            ],
            default_kwargs=variant,
        )
        for exp_id in range(num_hp_settings):
            variant = sweeper.generate_random_hyperparameters()
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
