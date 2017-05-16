"""
Use an oracle qfunction to train a policy in bptt-ddpg style.
"""
import joblib
from hyperopt import hp
import numpy as np
import tensorflow as tf
import random

from railrl.algos.ddpg import TargetUpdateMode
from railrl.data_management.ocm_subtraj_replay_buffer import (
    OcmSubtrajReplayBuffer
)
from railrl.data_management.updatable_subtraj_replay_buffer import \
    UpdatableSubtrajReplayBuffer
from railrl.envs.memory.one_char_memory import (
    OneCharMemoryEndOnly,
)
from railrl.envs.memory.high_low import HighLow
from railrl.envs.water_maze import WaterMaze
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
    LstmLinearCell,
    LstmMlpCell,
    # LstmLinearCellNoiseAll,
    SeparateLstmLinearCell,
    DebugCell,
)
# from railrl.algos.writeback_bptt_ddpt import WritebackBpttDDPG
from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.algos.noop_algo import NoOpBpttDDPG
# from railrl.algos.sum_bptt_ddpg import SumBpttDDPG
from railrl.exploration_strategies.noop import NoopStrategy
from railrl.exploration_strategies.onehot_sampler import OneHotSampler
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy

from railrl.exploration_strategies.action_aware_memory_strategy import \
    ActionAwareMemoryStrategy
from railrl.launchers.launcher_util import (
    run_experiment_here,
    create_base_log_dir,
)
from railrl.misc.hypopt import optimize_and_save
# from railrl.qfunctions.memory.mlp_memory_qfunction import MlpMemoryQFunction


def run_ocm_experiment(variant):
    from railrl.algos.oracle_bptt_ddpg import OracleBpttDdpg
    from railrl.algos.meta_bptt_ddpg import MetaBpttDdpg
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
    from railrl.qfunctions.memory.hint_mlp_memory_qfunction import (
        HintMlpMemoryQFunction
    )
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
    meta_qf_params = variant['meta_qf_params']
    es_params = variant['es_params']
    replay_buffer_class = variant['replay_buffer_class']
    replay_buffer_params = variant['replay_buffer_params']
    replay_buffer_params['memory_dim'] = memory_dim

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
        **env_es_params
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
            num_env_obs_dims_to_use=1,
            **policy_params
        )

    ddpg_params = ddpg_params.copy()
    if oracle_mode == 'none':
        qf_params['use_time'] = False
        qf_params['use_target'] = False
        qf = HintMlpMemoryQFunction(
            name_or_scope="critic",
            hint_dim=env_action_dim,
            max_time=H,
            env_spec=env.spec,
            **qf_params
        )
        algo_class = variant['algo_class']
    elif oracle_mode == 'oracle' or oracle_mode == 'meta':
        oracle_params = variant['oracle_params']
        qf = qf or HintMlpMemoryQFunction(
            name_or_scope="hint_critic",
            hint_dim=env_action_dim,
            max_time=H,
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
        ddpg_params.update(oracle_params)
    else:
        raise Exception("Unknown mode: {}".format(oracle_mode))
    if oracle_mode == 'meta':
        meta_qf = HintMlpMemoryQFunction(
            name_or_scope="meta_critic",
            hint_dim=env_action_dim,
            max_time=H,
            env_spec=env.spec,
            **meta_qf_params
        )
        algo_class = MetaBpttDdpg
        meta_params = variant['meta_params']
        ddpg_params['meta_qf'] = meta_qf
        ddpg_params.update(meta_params)

    algorithm = algo_class(
        env=env,
        exploration_strategy=es,
        policy=policy,
        qf=qf,
        env_obs_dim=env_obs_dim,
        env_action_dim=env_action_dim,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_params=replay_buffer_params,
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
    num_hp_settings = 100

    # n_seeds = 10
    # mode = 'ec2'
    # exp_prefix = '5-16-learning-rate-num-grad-steps'
    # run_mode = 'grid'
    # version = 'reparam'

    """
    Miscellaneous Params
    """
    n_rollouts_per_epoch = 100
    n_rollouts_per_eval = 64
    oracle_mode = 'none'
    algo_class = BpttDDPG
    # algo_class = NoOpBpttDDPG
    load_policy_file = (
        '/home/vitchyr/git/rllab-rail/railrl/data/reference/expert'
        '/ocm_reward_magnitude5_H6_nbptt6_100p'
        '/params.pkl'
    )
    load_policy_file = None

    """
    Set all the hyperparameters!
    """
    # env_class = WaterMaze
    # env_class = OneCharMemoryEndOnly
    env_class = HighLow
    H = 32
    env_params = dict(
        num_steps=H,
        n=2,
        zero_observation=True,
        output_target_number=False,
        output_time=False,
        episode_boundary_flags=False,
        max_reward_magnitude=1,
    )

    epoch_length = H * n_rollouts_per_epoch
    eval_samples = H * n_rollouts_per_eval
    max_path_length = H + 2
    # noinspection PyTypeChecker
    ddpg_params = dict(
        batch_size=32,
        n_epochs=30,
        min_pool_size=128,
        replay_pool_size=(H+1)*1000,
        n_updates_per_time_step=1,
        # replay_pool_size=int(32*(H+1)*5/4),
        epoch_length=epoch_length,
        eval_samples=eval_samples,
        max_path_length=max_path_length,
        discount=1.0,
        save_tf_graph=False,
        # Target network
        soft_target_tau=0.01,
        hard_update_period=1000,
        target_update_mode=TargetUpdateMode.HARD,
        # QF hyperparameters
        qf_learning_rate=1e-3,
        num_extra_qf_updates=0,
        extra_qf_training_mode='fixed',
        extra_train_period=100,
        qf_weight_decay=0.01,
        qf_total_loss_tolerance=0.03,
        train_qf_on_all=False,
        # Policy hps
        policy_learning_rate=1e-3,
        max_num_q_updates=1000,
        train_policy=True,
        write_policy_learning_rate=1e-3,
        train_policy_on_all_qf_timesteps=False,
        # memory
        num_bptt_unrolls=32,
        bpt_bellman_error_weight=1,
        reward_low_bellman_error_weight=0.,
        saved_write_loss_weight=1,
    )

    # noinspection PyTypeChecker
    policy_params = dict(
        # rnn_cell_class=LstmLinearCell,
        rnn_cell_class=SeparateLstmLinearCell,
        # rnn_cell_class=LstmLinearCellNoiseAll,
        # rnn_cell_class=DebugCell,
        rnn_cell_params=dict(
            use_peepholes=True,
            env_noise_std=.0,
            memory_noise_std=0.,
            output_nonlinearity=tf.nn.tanh,
            env_hidden_sizes=[],
            # env_hidden_activation=tf.tanh,
        )
    )

    oracle_params = dict(
        env_grad_distance_weight=0.,
        write_grad_distance_weight=0.,
        qf_grad_mse_from_one_weight=0.,
        regress_onto_values_weight=0.,
        bellman_error_weight=1.,
        use_oracle_qf=False,
        unroll_through_target_policy=False,
    )

    meta_qf_params = dict(
        use_time=False,
        use_target=False,
    )
    meta_params = dict(
        meta_qf_learning_rate=0.0001900271829580542,
        meta_qf_output_weight=0,
        qf_output_weight=1,
    )

    # noinspection PyTypeChecker
    es_params = dict(
        # env_es_class=NoopStrategy,
        env_es_class=OUStrategy,
        env_es_params=dict(
            max_sigma=1,
            min_sigma=1,
            decay_period=epoch_length*15,
            softmax=True,
            laplace_weight=0.,
        ),
        # memory_es_class=NoopStrategy,
        memory_es_class=OUStrategy,
        memory_es_params=dict(
            max_sigma=1,
            min_sigma=1,
            decay_period=epoch_length*15,
            softmax=True,
        ),
        noise_action_to_memory=False,
    )

    # noinspection PyTypeChecker
    qf_params = dict(
        # hidden_nonlinearity=tf.nn.relu,
        # output_nonlinearity=tf.nn.tanh,
        # hidden_nonlinearity=tf.identity,
        # output_nonlinearity=tf.identity,
        # embedded_hidden_sizes=[100, 64, 32],
        # observation_hidden_sizes=[100],
        use_time=False,
        use_target=False,
        dropout_keep_prob=None,
    )

    memory_dim = 20
    replay_buffer_params = dict()

    """
    Create monolithic variant dictionary
    """
    # noinspection PyTypeChecker
    variant = dict(
        memory_dim=memory_dim,
        exp_prefix=exp_prefix,
        algo_class=algo_class,
        version=version,
        load_policy_file=load_policy_file,
        oracle_mode=oracle_mode,
        env_class=env_class,
        env_params=env_params,
        ddpg_params=ddpg_params,
        policy_params=policy_params,
        qf_params=qf_params,
        meta_qf_params=meta_qf_params,
        oracle_params=oracle_params,
        es_params=es_params,
        meta_params=meta_params,
        replay_buffer_class=OcmSubtrajReplayBuffer,
        # replay_buffer_class=UpdatableSubtrajReplayBuffer,
        replay_buffer_params=replay_buffer_params,
    )

    if run_mode == 'hyperopt':
        search_space = {
            'policy_params.rnn_cell_params.env_noise_std': hp.uniform(
                'policy_params.rnn_cell_params.env_noise_std',
                0.,
                5,
            ),
            'policy_params.rnn_cell_params.memory_noise_std': hp.uniform(
                'policy_params.rnn_cell_params.memory_noise_std',
                0.,
                5,
            ),
            'ddpg_params.bpt_bellman_error_weight': hp.loguniform(
                'ddpg_params.bpt_bellman_error_weight',
                np.log(0.01),
                np.log(1000),
            ),
            'ddpg_params.qf_learning_rate': hp.loguniform(
                'ddpg_params.qf_learning_rate',
                np.log(0.00001),
                np.log(0.01),
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
            # 'memory_dim': [2, 20, 100],
            # 'policy_params.rnn_cell_params.env_noise_std': [0., 0.2, 1.],
            # 'policy_params.rnn_cell_params.memory_noise_std': [0., 0.2, 1.],
            # 'policy_params.rnn_cell_params.env_hidden_sizes': [
            #     [],
            #     [32],
            #     [32, 32],
            # ],
            # 'ddpg_params.qf_weight_decay': [0, 0.001],
            # 'ddpg_params.reward_low_bellman_error_weight': [0, 0.1, 1., 10.],
            # 'ddpg_params.num_extra_qf_updates': [0, 5],
            # 'ddpg_params.batch_size': [32, 128],
            # 'ddpg_params.replay_pool_size': [900, 90000],
            # 'ddpg_params.num_bptt_unrolls': [8, 6, 5, 4, 2],
            'ddpg_params.n_updates_per_time_step': [1, 5, 10],
            'ddpg_params.policy_learning_rate': [1e-3, 1e-4, 1e-5],
            # 'ddpg_params.bpt_bellman_error_weight': [10],
            # 'ddpg_params.saved_write_loss_weight': [0, 1, 10],
            # 'qf_params.dropout_keep_prob': [0.5, None],
            # 'meta_params.meta_qf_learning_rate': [1e-3, 1e-4],
            # 'meta_params.meta_qf_output_weight': [0, 0.1, 5],
            # 'meta_params.qf_output_weight': [0, 1],
            # 'env_params.episode_boundary_flags': [True, False],
            # 'env_params.num_steps': [8, 10, 12],
            # 'es_params.memory_es_class': [GaussianStrategy, OUStrategy],
            # 'es_params.env_es_class': [GaussianStrategy, OUStrategy],
            # 'es_params.memory_es_params.max_sigma': [3, 1],
            # 'es_params.memory_es_params.min_sigma': [1],
            # 'es_params.env_es_params.max_sigma': [3, 1],
            # 'es_params.env_es_params.min_sigma': [1],
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
                env_es_class,
                memory_es_class,
                env_noise_std,
                memory_noise_std,
        ) in enumerate([
            ("Gaussian", OUStrategy, 0),
            ("Reparam", NoopStrategy, 0.2)
        ]):
            variant['version'] = version
            variant['es_params']['env_es_class'] = env_es_class
            variant['es_params']['memory_es_class'] = memory_es_class
            variant['policy_params']['rnn_cell_params']['env_noise_std'] = (
                env_noise_std
            )
            variant['policy_params']['rnn_cell_params']['memory_noise_std'] = (
                memory_noise_std
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
