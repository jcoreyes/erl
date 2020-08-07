import random

import rlkit.misc.hyperparameter as hyp
from rlkit.envs.multitask.discrete_reacher_2d import DiscreteReacher2D
from rlkit.envs.multitask.multitask_env import MultitaskToFlatEnv
from rlkit.envs.wrappers import ConvertEnvToRllab
from rlkit.launchers.launcher_util import run_experiment
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy


def experiment(variant):
    env = variant['env_class']()
    if variant['multitask']:
        env = MultitaskToFlatEnv(env)
    env = ConvertEnvToRllab(env)

    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        **variant['policy_kwargs'],
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(
        #     **optimizer_params
        # )),
        **variant['trpo_params']
    )
    algo.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-trpo-baseline"

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "trpo-reacher-discerete-baseline"

    num_steps_per_iteration = 100000
    H = 200  # For CartPole and MountainVar, the max length is 200
    num_iterations = 50
    # noinspection PyTypeChecker
    variant = dict(
        trpo_params=dict(
            batch_size=num_steps_per_iteration,
            max_path_length=H,
            n_itr=num_iterations,
            discount=.99,
            n_parallel=1,
            step_size=0.01,
        ),
        optimizer_params=dict(
            base_eps=1e-5,
        ),
        policy_kwargs=dict(
            hidden_sizes=(100, 100),
        ),
        multitask=False,
    )
    search_space = {
        'env_class': [
            DiscreteReacher2D,
            # MountainCar,
            # CartPole,
            # CartPoleAngleOnly,
        ],
        'multitask': [False, True],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
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
