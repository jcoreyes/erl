import random

# from rlkit.envs.multitask.half_cheetah import GoalXVelHalfCheetah
from rlkit.envs.multitask.ant_env import GoalXYPosAnt, GoalXYPosAndVelAnt
from rlkit.envs.multitask.half_cheetah import GoalXVelHalfCheetah, \
    GoalXPosHalfCheetah
from rlkit.envs.multitask.multitask_env import MultitaskToFlatEnv
from rlkit.envs.multitask.point2d_uwall import MultitaskPoint2dUWall
from rlkit.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from rlkit.envs.multitask.pusher3d import MultitaskPusher3DEnv
from rlkit.envs.multitask.reacher_7dof import (
    # Reacher7DofGoalStateEverything,
    Reacher7DofXyzGoalState,
    Reacher7DofXyzPosAndVelGoalState)
from rlkit.envs.multitask.walker2d_env import Walker2DTargetXPos
from rlkit.envs.wrappers import (
    ConvertEnvToTf, NormalizedBoxEnv
)
from rlkit.launchers.launcher_util import run_experiment
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import (
    ConjugateGradientOptimizer,
    FiniteDifferenceHvp,
)
import rlkit.misc.hyperparameter as hyp


def experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    if variant['multitask']:
        env = MultitaskToFlatEnv(env)
    env = NormalizedBoxEnv(env)
    env = ConvertEnvToTf(env)

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

    # n_seeds = 2
    # mode = "ec2"
    # exp_prefix = "reacher-target-pos-and-vel-done-when-hit"

    num_epochs = 100
    num_steps_per_epoch = 10000
    num_steps_per_eval = 1000
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
        env_kwargs=dict(),
        multitask=False,
        version="TRPO",
        algorithm="TRPO",
    )
    search_space = {
        'env_class': [
            # GoalXVelHalfCheetah,
            # GoalXPosHalfCheetah,
            # Reacher7DofXyzGoalState,
            # GoalXYPosAnt,
            # GoalXYPosAndVelAnt,
            # Reacher7DofXyzPosAndVelGoalState,
            # CylinderXYPusher2DEnv,
            # MultitaskPusher3DEnv,
            # Walker2DTargetXPos,
            MultitaskPoint2dUWall,
        ],
        # 'env_kwargs.max_speed': [
        #     0.03,
        # ],
        # 'env_kwargs.speed_weight': [
        #     0.99,
        # ],
        # 'env_kwargs.done_threshold': [
        #     0.01,
        # ],
        'multitask': [True],
        # 'algo_kwargs.step_size': [
        #     100, 1, 0.01, 0.0001,
        # ],
        'algo_kwargs.max_path_length': [
            100
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_id=exp_id,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
            )
