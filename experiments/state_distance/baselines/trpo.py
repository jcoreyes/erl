import random

from railrl.envs.multitask.multitask_env import MultitaskToFlatEnv
from railrl.envs.multitask.reacher_7dof import Reacher7DofAngleGoalState
from railrl.envs.wrappers import normalize_and_convert_to_tf_env
from railrl.launchers.launcher_util import run_experiment
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import (
    ConjugateGradientOptimizer,
    FiniteDifferenceHvp,
)
import railrl.misc.hyperparameter as hyp


def experiment(variant):
    env = variant['env_class']()
    env = normalize_and_convert_to_tf_env(env)
    if variant['multitask']:
        env = MultitaskToFlatEnv(env)

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
    mode = "local"
    exp_prefix = "dev-trpo-baseline"

    n_seeds = 5
    mode = "ec2"
    exp_prefix = "trpo-reacher-7dof-angles-only-2"

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
        multitask=False,
    )
    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            seed = random.randint(0, 999999)
            run_experiment(
                experiment,
                exp_id=exp_id,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                use_gpu=False,
                snapshot_mode='gap',
                snapshot_gap=5,
            )
