import random

import railrl.misc.hyperparameter as hyp
from railrl.envs.multitask.cartpole_env import CartPole, CartPoleAngleOnly
from railrl.envs.multitask.discrete_reacher_2d import DiscreteReacher2D
from railrl.envs.multitask.mountain_car_env import MountainCar
from railrl.envs.multitask.multitask_env import MultitaskToFlatEnv
from railrl.envs.wrappers import normalize_and_convert_to_rllab_env, \
    ConvertEnvToRllab
from railrl.launchers.launcher_util import run_experiment
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import convert_gym_space
from rllab.envs.normalized_env import normalize
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy


def experiment(variant):
    env = variant['env_class']()
    if variant['multitask']:
        env = MultitaskToFlatEnv(env)
    env = ConvertEnvToRllab(env)

    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
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

    # n_seeds = 3
    # mode = "ec2"
    # exp_prefix = "classic-env-trpo-baseline"

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
