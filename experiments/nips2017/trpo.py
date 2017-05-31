"""
TRPO
"""
from railrl.envs.water_maze import WaterMazeEasy, WaterMazeMemory
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
    seed = variant['seed']
    env_class = variant['env_class']
    env_params = variant['env_params']

    set_seed(seed)

    """
    Code for running the experiment.
    """

    env = env_class(**env_params)

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
    exp_prefix = "dev-trpo"

    n_seeds = 10
    mode = "ec2"
    exp_prefix = "5-30-benchmark-trpo-small-water-maze-memory-h50"

    H = 50
    # noinspection PyTypeChecker
    trpo_params = dict(
        batch_size=10000,
        max_path_length=1000,  # Environment should stop it
        n_itr=100,
        discount=1.,
        step_size=0.01,
    )
    optimizer_params = dict(
        base_eps=1e-5,
    )
    env_params = dict(
        num_steps=H,
        use_small_maze=True,
    )
    USE_EC2 = False
    exp_id = -1
    # noinspection PyTypeChecker
    variant = dict(
        H=H,
        exp_prefix=exp_prefix,
        trpo_params=trpo_params,
        optimizer_params=optimizer_params,
        version='TRPO',
        # env_class=HighLow,
        # env_class=WaterMazeEasy,
        env_class=WaterMazeMemory,
        env_params=env_params,
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
