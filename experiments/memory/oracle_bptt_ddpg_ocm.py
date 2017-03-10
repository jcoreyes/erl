"""
Use an oracle qfunction to train a policy in bptt-ddpg style.
"""
from itertools import product
from railrl.launchers.launcher_util import (
    run_experiment,
)


def run_linear_ocm_exp(variant):
    from railrl.algos.oracle_bptt_ddpg import OracleBpttDDPG
    from railrl.qfunctions.memory.oracle_qfunction import OracleQFunction
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

    """
    Set up experiment variants.
    """
    H = variant['H']
    seed = variant['seed']
    num_values = variant['num_values']
    ddpg_params = variant['ddpg_params']

    env_action_dim = num_values + 1
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
    )

    es = ProductStrategy([OneHotSampler(), NoopStrategy()])
    qf = OracleQFunction(
        name_or_scope="oracle_critic",
        env=env,
        env_spec=env.spec,
    )
    algorithm = OracleBpttDDPG(
        env,
        es,
        policy,
        qf,
        env_obs_dim=env_action_dim,
        **ddpg_params
    )

    algorithm.train()


if __name__ == '__main__':
    n_seed = 1
    # exp_prefix = "3-8-bptt-ddpg-ocm-benchmark"
    exp_prefix = "dev-oracle-bptt-ddpg"

    """
    DDPG Params
    """
    n_batches_per_epoch = 100
    n_batches_per_eval = 64
    batch_size = 32
    n_epochs = 100
    lstm_state_size = 10
    min_pool_size = 100
    replay_pool_size = 100000

    mode = 'here'
    exp_id = -1
    for H, num_values, num_bptt_unrolls in product(
        [8],
        [4],
        [4],
    ):
        print("H", H)
        print("num_values", num_values)
        print("num_bptt_unrolls", num_bptt_unrolls)
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
            # qf_learning_rate=1e-1,
            # policy_learning_rate=1e-1,
        )
        variant = dict(
            H=H,
            num_values=num_values,
            exp_prefix=exp_prefix,
            ddpg_params=ddpg_params,
            lstm_state_size=lstm_state_size,
        )
        for seed in range(n_seed):
            run_experiment(
                run_linear_ocm_exp,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
        )