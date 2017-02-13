"""
Check softmax_memory_policy on OneCharMemoryEndOnly task.
"""
from railrl.launchers.launcher_util import (
    run_experiment,
    run_experiment_here,
)


def run_linear_ocm_exp(variant):
    from railrl.algos.ddpg_ocm import DdpgOcm
    from railrl.qfunctions.memory_qfunction import MemoryQFunction
    from railrl.exploration_strategies.noop import NoopStrategy
    from railrl.envs.memory.continuous_memory_augmented import (
        ContinuousMemoryAugmented
    )
    from railrl.envs.memory.one_char_memory import OneCharMemoryEndOnly
    from railrl.policies.memory.softmax_memory_policy import SoftmaxMemoryPolicy
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

    onehot_dim = num_values + 1
    set_seed(seed)

    """
    Code for running the experiment.
    """

    env = OneCharMemoryEndOnly(n=num_values, num_steps=H)
    env = ContinuousMemoryAugmented(
        env,
        num_memory_states=onehot_dim,
    )

    policy = SoftmaxMemoryPolicy(
        name_or_scope="policy",
        memory_and_action_dim=onehot_dim,
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

    algorithm.train()


if __name__ == '__main__':
    n_seeds = 1
    exp_prefix = "2-12-dev-softmax-memory-policy-ocm"

    """
    DDPG Params
    """
    n_batches_per_epoch = 100
    n_batches_per_eval = 100
    batch_size = 64
    n_epochs = 100
    replay_pool_size = 100

    USE_EC2 = False
    exp_count = -1
    for H in [2]:
        for num_values in [2]:
            print("H", H)
            print("num_values", num_values)
            exp_count += 1
            min_pool_size = 10 * H
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
                H=H,
                num_values=num_values,
                exp_prefix=exp_prefix,
                ddpg_params=ddpg_params,
            )
            for seed in range(n_seeds):
                variant['seed'] = seed
                variant['exp_count'] = exp_count

                if USE_EC2:
                    run_experiment(
                        run_linear_ocm_exp,
                        exp_prefix=exp_prefix,
                        seed=seed,
                        mode="ec2",
                        variant=variant,
                    )
                else:
                    run_experiment_here(
                        run_linear_ocm_exp,
                        exp_prefix=exp_prefix,
                        variant=variant,
                        exp_count=exp_count,
                        seed=seed,
                    )
