import random

# from railrl.envs.multitask.half_cheetah import GoalXVelHalfCheetah
from railrl.envs.multitask.ant_env import GoalXYPosAnt
from railrl.envs.multitask.half_cheetah import GoalXVelHalfCheetah, \
    GoalXPosHalfCheetah
from railrl.envs.multitask.multitask_env import MultitaskToFlatEnv
from railrl.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from railrl.envs.multitask.pusher3d import MultitaskPusher3DEnv
from railrl.envs.multitask.reacher_7dof import (
    # Reacher7DofGoalStateEverything,
    Reacher7DofXyzGoalState,
)
from railrl.envs.multitask.walker2d_env import Walker2DTargetXPos
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
    env = variant['env_class'](**variant['env_kwargs'])
    if variant['multitask']:
        env = MultitaskToFlatEnv(env)
    env = normalize_and_convert_to_tf_env(env)

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        **variant['policy_params']
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    optimizer_params = variant['optimizer_params']
    algo_kwargs = variant['algo_kwargs']
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(
            **optimizer_params
        )),
        **algo_kwargs
    )
    algo.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-state-distance-trpo-baseline"

    n_seeds = 1
    mode = "ec2"
    exp_prefix = "normal-gear-ratio-ant-d6"

    num_epochs = 1000
    num_steps_per_epoch = 10000
    num_steps_per_eval = 10000
    max_path_length = 100

    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            batch_size=num_steps_per_epoch,
            max_path_length=max_path_length,
            n_itr=num_epochs,
            discount=.99,
            step_size=0.01,
        ),
        optimizer_params=dict(
            base_eps=1e-5,
        ),
        policy_params=dict(
            hidden_sizes=(300, 300),
        ),
        multitask=False,
        version="TRPO",
        algorithm="TRPO",
    )
    search_space = {
        'env_class': [
            # GoalXVelHalfCheetah,
            # GoalXPosHalfCheetah,
            # Reacher7DofXyzGoalState,
            GoalXYPosAnt,
            # CylinderXYPusher2DEnv,
            # MultitaskPusher3DEnv,
            # Walker2DTargetXPos,
        ],
        'env_kwargs.max_distance': [
            6,
        ],
        'env_kwargs.use_low_gear_ratio': [
            False,
        ],
        'multitask': [True],
        'algo_kwargs.step_size': [
            1, 0.01, 0.0001,
        ],
        'algo_kwargs.max_path_length': [
            50, 100
        ],
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
            )
