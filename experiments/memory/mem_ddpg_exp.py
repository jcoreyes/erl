"""
Check DDPG + memory states on OneCharMemory task.
"""
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import (
    run_experiment,
    set_seed,
)
from rlkit.qfunctions.memory.mlp_memory_qfunction import MlpMemoryQFunction
from rlkit.tf.policies.memory.action_aware_memory_policy import \
    ActionAwareMemoryPolicy


def run_linear_ocm_exp(variant):
    from rlkit.tf.ddpg import DDPG
    from rlkit.envs.memory.continuous_memory_augmented import (
        ContinuousMemoryAugmented
    )
    from rlkit.envs.memory.one_char_memory import (
        OneCharMemoryEndOnly,
    )
    from rlkit.launchers.launcher_util import (
        set_seed,
    )

    """
    Set up experiment variants.
    """
    H = variant['H']
    seed = variant['seed']
    num_values = variant['num_values']
    algo_params = variant['algo_params']

    set_seed(seed)
    onehot_dim = num_values + 1

    env_action_dim = num_values + 1

    """
    Code for running the experiment.
    """

    env = OneCharMemoryEndOnly(n=num_values, num_steps=H, softmax_action=True)
    env = ContinuousMemoryAugmented(
        env,
        num_memory_states=onehot_dim,
    )
    # env = FlattenedProductBox(env)

    # qf = FeedForwardCritic(
    #     name_or_scope="critic",
    #     env_spec=env.spec,
    # )
    qf = MlpMemoryQFunction(
        name_or_scope="critic",
        env_spec=env.spec,
    )
    policy = ActionAwareMemoryPolicy(
        name_or_scope="noisy_policy",
        action_dim=env_action_dim,
        memory_dim=memory_dim,
        env_spec=env.spec,
    )
    es = OUStrategy(env_spec=env.spec)
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
        **algo_params
    )

    algorithm.train()


if __name__ == '__main__':
    n_seeds = 1
    exp_prefix = "dev-5-1-no=bptt"

    n_batches_per_epoch = 100
    n_batches_per_eval = 64
    batch_size = 32
    n_epochs = 100
    memory_dim = 20
    # memory_dim = 4
    # min_pool_size = 10*max(n_batches_per_epoch, batch_size)
    min_pool_size = max(n_batches_per_epoch, batch_size)
    replay_pool_size = 100000

    USE_EC2 = False
    exp_id = -1
    for H in [8]:
        for num_values in [2]:
            epoch_length = H * n_batches_per_epoch
            eval_samples = H * n_batches_per_eval
            max_path_length = H + 2
            algo_params = dict(
                batch_size=batch_size,
                n_epochs=n_epochs,
                min_pool_size=min_pool_size,
                replay_pool_size=replay_pool_size,
                epoch_length=epoch_length,
                eval_samples=eval_samples,
                max_path_length=max_path_length,
                discount=1.0,
            )
            print("H", H)
            print("num_values", num_values)
            variant = dict(
                H=H,
                num_values=num_values,
                exp_prefix=exp_prefix,
                algo_params=algo_params,
            )
            for seed in range(n_seeds):
                exp_id += 1
                set_seed(seed)
                variant['seed'] = seed
                variant['exp_id'] = exp_id

                if USE_EC2:
                    mode = "ec2"
                else:
                    mode = "here"
                run_experiment(
                    run_linear_ocm_exp,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                )
