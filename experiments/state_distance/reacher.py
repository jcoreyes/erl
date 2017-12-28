import random

from railrl.envs.multitask.reacher_env import XyMultitaskReacherEnv
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.simple import ZeroPolicy
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.state_distance.state_distance_q_learning import (
    StateDistanceQLearning
)


def experiment(variant):
    env = XyMultitaskReacherEnv()
    # env = gym_env("Reacher-v1")
    # env = convert_to_tf_env(env)
    # env = normalize(env)
    action_space = convert_gym_space(env.action_space)
    observation_space = convert_gym_space(env.observation_space)
    es = OUStrategy(action_space=action_space)
    qf = FeedForwardQFunction(
        int(observation_space.flat_dim) + env.goal_dim,
        int(action_space.flat_dim),
        400,
        300,
    )
    exploration_policy = ZeroPolicy(
        int(action_space.flat_dim),
    )
    policy = FeedForwardPolicy(
        int(observation_space.flat_dim) + env.goal_dim,
        int(action_space.flat_dim),
        400,
        300,
    )
    algorithm = StateDistanceQLearning(
        env,
        exploration_strategy=es,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        # exploration_policy=policy,
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "here"
    exp_prefix = "7-9-dev-state-distance-reacher"

    # n_seeds = 10
    # mode = "ec2"
    # exp_prefix = "7-9-state-distance-ddpg-baseline"

    num_steps_per_iteration = 1000
    num_steps_per_eval = 1000
    H = 1000
    num_iterations = 100
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=num_iterations,
            num_steps_per_epoch=num_steps_per_iteration,
            num_steps_per_eval=num_steps_per_eval,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=H,
            discount=0.,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
        version="DDPG",
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
        )
