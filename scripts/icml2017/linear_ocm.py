from railrl.launchers.launcher_util import (
    run_experiment,
    reset_execution_environment,
)


def run_linear_ocm_exp(variant):
    from railrl.algos.ddpg_ocm import DdpgOcm
    from railrl.qfunctions.memory_qfunction import MemoryQFunction
    from railrl.exploration_strategies.noop import NoopStrategy
    from railrl.envs.memory.continuous_memory_augmented import (
        ContinuousMemoryAugmented
    )
    from railrl.envs.memory.one_char_memory import OneCharMemoryEndOnly
    from railrl.policies.linear_ocm_policy import LinearOcmPolicy
    from railrl.launchers.launcher_util import (
        setup_logger,
        set_seed,
    )

    """
    Set up experiment variants.
    """
    H = variant['H']
    seed = variant['seed']
    exp_prefix = variant['exp_prefix']
    exp_count = variant['exp_count']

    num_values = 2
    batch_size = 64
    n_epochs = 5
    min_pool_size = 10 * H
    replay_pool_size = 1000

    n_batches_per_epoch = 100
    n_batches_per_eval = 100

    set_seed(seed)

    epoch_length = H * n_batches_per_epoch
    eval_samples = H * n_batches_per_eval
    max_path_length = H + 1

    ddpg_params = dict(
        batch_size=batch_size,
        n_epochs=n_epochs,
        min_pool_size=min_pool_size,
        replay_pool_size=replay_pool_size,
        epoch_length=epoch_length,
        eval_samples=eval_samples,
        max_path_length=max_path_length,
    )
    variant = dict(
        num_values=num_values,
        H=H,
        ddpg_params=ddpg_params,
    )

    """
    Code for running the experiment.
    """

    onehot_dim = num_values + 1

    env = OneCharMemoryEndOnly(n=num_values, num_steps=H)
    env = ContinuousMemoryAugmented(
        env,
        num_memory_states=onehot_dim,
    )

    policy = LinearOcmPolicy(
        name_or_scope="policy",
        memory_and_action_dim=onehot_dim,
        horizon=H,
        env_spec=env.spec,
    )

    es = NoopStrategy()
    qf = MemoryQFunction(
        name_or_scope="critic",
        env_spec=env.spec,
    )
    algorithm = DdpgOcm(
        env,
        es,
        policy,
        qf,
        **ddpg_params
    )

    setup_logger(
        exp_prefix=exp_prefix,
        variant=variant,
        exp_count=exp_count,
    )
    algorithm.train()


if __name__ == '__main__':
    n_seeds = 3
    H = 2
    exp_prefix = "2-12-test_linear_ocm"
    USE_EC2 = False
    for seed in range(n_seeds):
        variant = dict(
            H=H,
            seed=seed,
            exp_count=seed,
            exp_prefix=exp_prefix,
        )

        if USE_EC2:
            run_experiment(
                run_linear_ocm_exp,
                exp_prefix=exp_prefix,
                seed=seed,
                mode="ec2",
                variant=variant,
            )
        else:
            run_linear_ocm_exp(variant)
            reset_execution_environment()
