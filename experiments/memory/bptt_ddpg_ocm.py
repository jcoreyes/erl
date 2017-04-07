"""
Use an oracle qfunction to train a policy in bptt-ddpg style.
"""
from itertools import product
import tensorflow as tf
import random

from railrl.launchers.launcher_util import (
    run_experiment,
)
from railrl.policies.memory.lstm_memory_policy import (
    OutputAwareLstmCell,
    LstmLinearCell,
    FrozenHiddenLstmLinearCell,
    IRnnCell,
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
    from os.path import exists

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
    load_policy_file = variant.get('load_policy_file', None)

    env_action_dim = num_values + 1
    env_obs_dim = env_action_dim
    memory_dim = variant['memory_dim']
    set_seed(seed)

    """
    Code for running the experiment.
    """

    env = OneCharMemoryEndOnly(n=num_values, num_steps=H)
    env = ContinuousMemoryAugmented(
        env,
        num_memory_states=memory_dim,
    )
    if load_policy_file is None or not exists(load_policy_file):
        policy = LstmMemoryPolicy(
            name_or_scope="policy",
            action_dim=env_action_dim,
            memory_dim=memory_dim,
            env_spec=env.spec,
            **policy_params
        )
    else:
        import joblib
        with tf.Session():
            data = joblib.load(load_policy_file)
            policy = data['policy']
    es = ProductStrategy([OneHotSampler(), NoopStrategy()])

    ddpg_params = ddpg_params.copy()
    unroll_through_target_policy = ddpg_params.pop(
        'unroll_through_target_policy',
        False,
    )
    qf_tolerance = ddpg_params.pop('qf_tolerance', 1e-3)
    max_num_q_updates = ddpg_params.pop('max_num_q_updates', 10)
    train_policy = ddpg_params.pop('train_policy', 10)
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
        # qf = MlpMemoryQFunction(
        #     name_or_scope="critic",
        #     env_spec=env.spec,
        # )

        qf = HintMlpMemoryQFunction(
            name_or_scope="hint_critic",
            hint_dim=env_action_dim,
            env_spec=env.spec,
            # hidden_nonlinearity=tf.nn.tanh,
            # embedded_hidden_sizes=(32, 32),
            # observation_hidden_sizes=(32,),
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
        ddpg_params['qf_tolerance'] = qf_tolerance
        ddpg_params['max_num_q_updates'] = max_num_q_updates
        ddpg_params['train_policy'] = train_policy
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


if __name__ == '__main__':
    mode = 'here'
    n_seed = 1
    exp_prefix = "dev-4-7-bptt-ddpg-ocm-regress"
    version = 'dev'

    """
    DDPG Params
    """
    n_batches_per_epoch = 100
    n_batches_per_eval = 64
    batch_size = 32
    n_epochs = 100
    memory_dim = 20
    min_pool_size = max(n_batches_per_epoch, batch_size)
    replay_pool_size = 100000

    """
    Algorithm Selection
    """
    oracle_mode = 'regress'
    algo_class = BpttDDPG
    freeze_hidden = False
    unroll_through_target_policy = False

    """
    Policy Params
    """
    policy_rnn_cell_class = OutputAwareLstmCell
    # policy_rnn_cell_class = IRnnCell
    # policy_rnn_cell_class = LstmLinearCell
    # policy_rnn_cell_class = FrozenHiddenLstmLinearCell
    load_policy_file = (
        '/home/vitchyr/git/rllab-rail/railrl/data/reference/expert'
        '/ocm_66p'
        '/params.pkl'
    )
    # load_policy_file = None
    load_policy_file = 'none'

    exp_id = -1

    # top: H = 6, memory dim = 100
    # bottom: H = 6, memory dim = 20
    H = 5
    num_values = 2
    num_bptt_unrolls = 4
    num_extra_qf_updates = 0
    qf_learning_rate = 1e-4
    qf_tolerance = 1e-2
    policy_learning_rate = 5e-5
    max_num_q_updates = 2
    train_policy = True


    exp_id += 1
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
        qf_learning_rate=qf_learning_rate,
        policy_learning_rate=policy_learning_rate,
        discount=1.0,
        qf_tolerance=qf_tolerance,
        max_num_q_updates=max_num_q_updates,
        train_policy=train_policy,
        # soft_target_tau=1.0,
        # policy_learning_rate=1e-1,
    )
    policy_params = dict(
        rnn_cell_class=policy_rnn_cell_class,
    )
    variant = dict(
        H=H,
        num_values=num_values,
        exp_prefix=exp_prefix,
        ddpg_params=ddpg_params,
        policy_params=policy_params,
        memory_dim=memory_dim,
        oracle_mode=oracle_mode,
        algo_class=algo_class,
        freeze_hidden=freeze_hidden,
        version=version,
        load_policy_file=load_policy_file,
    )
    for _ in range(n_seed):
        seed = random.randint(0, 10000)
        run_experiment(
            run_ocm_experiment,
            exp_prefix=exp_prefix,
            seed=seed,
            mode=mode,
            variant=variant,
            exp_id=exp_id,
        )
