"""
TRPO + memory states.
"""
from railrl.envs.flattened_product_box import FlattenedProductBox
from railrl.envs.memory.continuous_memory_augmented import \
    ContinuousMemoryAugmented
from railrl.envs.memory.high_low import HighLow
from railrl.launchers.launcher_util import (
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
    from railrl.launchers.launcher_util import (
        set_seed,
    )

    """
    Set up experiment variants.
    """
    H = variant['H']
    seed = variant['seed']
    env_class = variant['env_class']
    memory_dim = variant['memory_dim']

    set_seed(seed)

    """
    Code for running the experiment.
    """

    env = env_class(num_steps=H)
    env = ContinuousMemoryAugmented(
        env,
        num_memory_states=memory_dim,
    )
    env = FlattenedProductBox(env)

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
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
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-mtrpo"

    n_seeds = 10
    mode = "ec2"
    exp_prefix = "5-17-benchmark-mtrpo-hl"

    # noinspection PyTypeChecker
    trpo_params = dict(
        batch_size=1000,
        max_path_length=1000,  # Environment should stop it
        n_itr=100,
        discount=1.,
        step_size=0.01,
    )
    optimizer_params = dict(
        base_eps=1e-5,
    )
    USE_EC2 = False
    exp_id = -1
    # noinspection PyTypeChecker
    variant = dict(
        H=32,
        exp_prefix=exp_prefix,
        trpo_params=trpo_params,
        optimizer_params=optimizer_params,
        version='Memory States + TRPO',
        env_class=HighLow,
        memory_dim=20,
    )
    for seed in range(n_seeds):
        exp_id += 1
        set_seed(seed)
        variant['seed'] = seed
        variant['exp_id'] = exp_id

        run_experiment(
            run_linear_ocm_exp,
            exp_prefix=exp_prefix,
            seed=seed,
            mode=mode,
            variant=variant,
            exp_id=exp_id,
        )
