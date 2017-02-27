"""
Check TRPO on OneCharMemory task.
"""
from railrl.envs.flattened_product_box import FlattenedProductBox
from railrl.launchers.launcher_util import (
    run_experiment,
    run_experiment_here,
)


def run_linear_ocm_exp(variant):
    import tensorflow as tf

    from sandbox.rocky.tf.algos.trpo import TRPO
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
    from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import (
        ConjugateGradientOptimizer,
        FiniteDifferenceHvp,
    )
    from railrl.envs.memory.continuous_memory_augmented import (
        ContinuousMemoryAugmented
    )
    from railrl.envs.memory.one_char_memory import (
        OneCharMemoryEndOnly,
        OneCharMemoryEndOnlyDiscrete,
    )
    from railrl.launchers.launcher_util import (
        set_seed,
    )
    from railrl.policies.rllab.deterministic_mlp_policy import (
        DeterministicMLPPolicy
    )

    """
    Set up experiment variants.
    """
    H = variant['H']
    seed = variant['seed']
    num_values = variant['num_values']

    onehot_dim = num_values + 1
    set_seed(seed)

    """
    Code for running the experiment.
    """

    env = OneCharMemoryEndOnly(n=num_values, num_steps=H, softmax_action=True)
    env = ContinuousMemoryAugmented(
        env,
        num_memory_states=onehot_dim,
    )
    env = FlattenedProductBox(env)

    # policy = CategoricalMLPPolicy(
    #     name="policy",
    #     env_spec=env.spec,
    #     # The neural network policy should have two hidden layers, each with 32 hidden units.
    #     hidden_sizes=(32, 32),
    # )
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32),
    )

    # policy = DeterministicMLPPolicy(
    #     name="policy",
    #     env_spec=env.spec,
    #     # The neural network policy should have two hidden layers, each with 32 hidden units.
    #     hidden_sizes=(32, 32),
    #     output_nonlinearity=tf.nn.softmax,
    # )
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    optimizer_params = variant['optimizer_params']
    trpo_params = variant['trpo_params']
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(
            **optimizer_params
        )),
        **trpo_params
    )

    algo.train()


if __name__ == '__main__':
    n_seeds = 1
    exp_prefix = "dev-ocm-trpo"
    """
    DDPG Params
    """
    n_batches_per_epoch = 100
    n_batches_per_eval = 100
    batch_size = 64
    n_epochs = 100

    USE_EC2 = False
    exp_id = -1
    for H in [4]:
        for num_values in [4]:
            print("H", H)
            print("num_values", num_values)
            exp_id += 1
            min_pool_size = H * 10
            replay_pool_size = 16 * H
            epoch_length = H * n_batches_per_epoch
            eval_samples = H * n_batches_per_eval
            max_path_length = H + 1
            trpo_params = dict(
                batch_size=4000,
                max_path_length=100,
                n_itr=40,
                discount=0.99,
                step_size=0.01,
            )
            optimizer_params = dict(
                base_eps=1e-5,
            )
            variant = dict(
                H=H,
                num_values=num_values,
                exp_prefix=exp_prefix,
                trpo_params=trpo_params,
                optimizer_params=optimizer_params,
            )
            for seed in range(n_seeds):
                variant['seed'] = seed
                variant['exp_id'] = exp_id

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
                        exp_id=exp_id,
                        seed=seed,
                    )
