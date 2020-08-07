"""
Check TRPO on OneCharMemory task.
"""
from rlkit.launchers.launcher_util import (
    run_experiment,
    set_seed,
)


def run_linear_ocm_exp(variant):
    from sandbox.rocky.tf.algos.trpo import TRPO
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import (
        ConjugateGradientOptimizer,
        FiniteDifferenceHvp,
    )
    from rlkit.envs.flattened_product_box import FlattenedProductBox
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

    set_seed(seed)
    onehot_dim = num_values + 1

    """
    Code for running the experiment.
    """

    env = OneCharMemoryEndOnly(n=num_values, num_steps=H, softmax_action=True)
    env = ContinuousMemoryAugmented(
        env,
        num_memory_states=onehot_dim,
    )
    env = FlattenedProductBox(env)

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32),
    )

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
    n_seeds = 5
    exp_prefix = "4-30-comparison-short"

    n_batches_per_itr = 100

    trpo_params = dict(
        batch_size=1000,
        max_path_length=100,
        n_itr=20,
        discount=0.99,
        step_size=0.01,
    )
    optimizer_params = dict(
        base_eps=1e-5,
    )
    USE_EC2 = False
    exp_id = -1
    for H in [8, 16]:
        for num_values in [2]:
            print("H", H)
            print("num_values", num_values)
            variant = dict(
                H=H,
                num_values=num_values,
                exp_prefix=exp_prefix,
                trpo_params=trpo_params,
                optimizer_params=optimizer_params,
                version='trpo_memory_10x_slow',
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
