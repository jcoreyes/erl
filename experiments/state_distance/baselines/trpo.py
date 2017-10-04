import random

from gym.envs.mujoco import PusherEnv

from railrl.envs.wrappers import normalize_and_convert_to_tf_env
from railrl.launchers.launcher_util import run_experiment
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import (
    ConjugateGradientOptimizer,
    FiniteDifferenceHvp,
)


def experiment(variant):
    env = PusherEnv()
    env = normalize_and_convert_to_tf_env(env)

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=(400, 300)
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

if __name__ == "__main__":
    n_seeds = 1
    mode = "here"
    exp_prefix = "pusher-env-trpo-baseline"

    # n_seeds = 3
    # mode = "ec2"
    # exp_prefix = "pusher-env-trpo-baseline"

    num_steps_per_iteration = 100000
    H = 1000
    num_iterations = 100
    # noinspection PyTypeChecker
    variant = dict(
        trpo_params=dict(
            batch_size=num_steps_per_iteration,
            max_path_length=H,  # Environment should stop it
            n_itr=num_iterations,
            discount=1.,
            step_size=0.01,
        ),
        optimizer_params=dict(
            base_eps=1e-5,
        ),
        version="TRPO",
    )
    for _ in range(n_seeds):
        seed = random.randint(0, 999999)
        run_experiment(
            experiment,
            exp_prefix=exp_prefix,
            seed=seed,
            mode=mode,
            variant=variant,
            use_gpu=False,
            snapshot_mode='gap',
            snapshot_gap=5,
        )
