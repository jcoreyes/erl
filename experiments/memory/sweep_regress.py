"""
Sweep hyperparameters to find good settings for RegressQBpttDdpg.
"""
import numpy as np

from railrl.misc.hypopt import optimize_and_save
from hyperopt import hp
from railrl.launchers.launcher_util import (
    run_experiment_here,
    create_base_log_dir,
)
from railrl.policies.memory.lstm_memory_policy import (
    OutputAwareLstmCell,
    LstmLinearCell,
    FrozenHiddenLstmLinearCell,
)
from railrl.algos.writeback_bptt_ddpt import WritebackBpttDDPG
from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.algos.sum_bptt_ddpg import SumBpttDDPG


def run_ocm_experiment(variant):
    from railrl.algos.oracle_bptt_ddpg import (
        OracleBpttDDPG,
        OracleUnrollBpttDDPG,
        RegressQBpttDdpg,
    )
    from railrl.qfunctions.memory.oracle_qfunction import OracleQFunction
    from railrl.qfunctions.memory.oracle_unroll_qfunction import (
        OracleUnrollQFunction
    )
    from railrl.exploration_strategies.noop import NoopStrategy
    from railrl.exploration_strategies.onehot_sampler import OneHotSampler
    from railrl.exploration_strategies.product_strategy import ProductStrategy
    from railrl.envs.memory.continuous_memory_augmented import (
        ContinuousMemoryAugmented
    )
    from railrl.envs.memory.one_char_memory import OneCharMemoryEndOnly
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

    """
    Set up experiment variants.
    """
    H = variant['H']
    seed = variant['seed']
    num_values = variant['num_values']
    ddpg_params = variant['ddpg_params']
    policy_params = variant['policy_params']
    num_bptt_unrolls = ddpg_params['num_bptt_unrolls']
    oracle_mode = variant['oracle_mode']

    env_action_dim = num_values + 1
    env_obs_dim = env_action_dim
    lstm_state_size = variant['lstm_state_size']
    memory_dim = 2 * lstm_state_size
    set_seed(seed)

    """
    Code for running the experiment.
    """

    env = OneCharMemoryEndOnly(n=num_values, num_steps=H)
    env = ContinuousMemoryAugmented(
        env,
        num_memory_states=memory_dim,
    )
    policy = LstmMemoryPolicy(
        name_or_scope="policy",
        action_dim=env_action_dim,
        memory_dim=memory_dim,
        env_spec=env.spec,
        **policy_params
    )
    es = ProductStrategy([OneHotSampler(), NoopStrategy()])

    if oracle_mode == 'none':
        qf = MlpMemoryQFunction(
            name_or_scope="critic",
            env_spec=env.spec,
        )
        algo_class = variant['algo_class']
        ddpg_params = ddpg_params.copy()
        ddpg_params.pop('unroll_through_target_policy')
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
    elif oracle_mode == 'hint':
        qf = HintMlpMemoryQFunction(
            name_or_scope="hint_critic",
            hint_dim=env_action_dim,
            env_spec=env.spec,
        )
        algo_class = OracleBpttDDPG
        ddpg_params = ddpg_params.copy()
        ddpg_params.pop('unroll_through_target_policy')
    elif oracle_mode == 'oracle':
        qf = OracleQFunction(
            name_or_scope="oracle_critic",
            env=env,
            env_spec=env.spec,
        )
        algo_class = OracleBpttDDPG
        ddpg_params = ddpg_params.copy()
        ddpg_params.pop('unroll_through_target_policy')
    elif oracle_mode == 'regress':
        # qf = MlpMemoryQFunction(
        #     name_or_scope="critic",
        #     env_spec=env.spec,
        # )
        qf = HintMlpMemoryQFunction(
            name_or_scope="hint_critic",
            hint_dim=env_action_dim,
            env_spec=env.spec,
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
        ddpg_params = ddpg_params.copy()
        ddpg_params.pop('unroll_through_target_policy')
        ddpg_params['oracle_qf'] = oracle_qf
    else:
        raise Exception("Unknown mode: {}".format(oracle_mode))

    algorithm = algo_class(
        env,
        es,
        policy,
        qf,
        env_obs_dim=env_obs_dim,
        replay_buffer_class=OcmSubtrajReplayBuffer,
        **ddpg_params
    )

    algorithm.train()
    return algorithm.final_score


if __name__ == '__main__':
    mode = 'here'
    n_seed = 1
    exp_prefix = "check-dev-sweep-regress"
    version = 'dev'

    """
    DDPG Params
    """
    n_batches_per_epoch = 100
    n_batches_per_eval = 64
    batch_size = 32
    n_epochs = 5
    lstm_state_size = 10
    min_pool_size = n_batches_per_epoch
    replay_pool_size = 100000
    num_extra_qf_updates = 4

    """
    Algorithm Selection
    """
    oracle_mode = 'regress'
    algo_class = BpttDDPG
    # algo_class = WritebackBpttDDPG
    # algo_class = SumBpttDDPG
    freeze_hidden = False
    unroll_through_target_policy = False

    H = 4
    num_values = 2
    num_bptt_unrolls = 4
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
        num_extra_qf_updates=num_extra_qf_updates,
    )
    variant = dict(
        H=H,
        num_values=num_values,
        exp_prefix=exp_prefix,
        ddpg_params=ddpg_params,
        lstm_state_size=lstm_state_size,
        # oracle_mode=oracle_mode,
        # algo_class=algo_class,
        freeze_hidden=freeze_hidden,
        version=version,
    )
    search_space = {
        # 'ddpg_params.qf_learning_rate': hp.loguniform('qf_learning_rate',
        #                                   np.log(1e-5),
        #                                   np.log(1e-2)),
        'policy_params.rnn_cell_class': hp.choice(
            'policy_params.rnn_cell_class',
            [
                OutputAwareLstmCell,
                LstmLinearCell,
                FrozenHiddenLstmLinearCell,
            ]
        ),
        'algo_class': hp.choice(
            'algo_class',
            [
                BpttDDPG,
                WritebackBpttDDPG,
                SumBpttDDPG,
            ]
        ),
        'oracle_mode': hp.choice(
            'oracle_mode',
            [
                'none',
                'unroll',
            ]
        ),
    }

    base_log_dir = create_base_log_dir(exp_prefix=exp_prefix)
    exp_id = -1
    def run_experiment_wrapper(hyperparams):
        global exp_id
        exp_id += 1
        return run_experiment_here(
            run_ocm_experiment,
            exp_prefix=exp_prefix,
            variant=hyperparams,
            exp_id=exp_id,
            seed=0,
        )
    optimize_and_save(
        base_log_dir,
        run_experiment_wrapper,
        search_space=search_space,
        extra_function_kwargs=variant,
        maximize=True,
        verbose=True,
        load_trials=True,
        num_rounds=10,
        num_evals_per_round=1,
    )
