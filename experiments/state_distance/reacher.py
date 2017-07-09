import random

from railrl.algos.qlearning.state_distance_q_learning import (
    StateDistanceQLearning
)
from railrl.envs.env_utils import gym_env
from railrl.envs.multitask.reacher_env import MultitaskReacherEnv
from railrl.envs.wrappers import convert_to_tf_env
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.policies.zero_policy import ZeroPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG

from rllab.envs.normalized_env import normalize
from gym.envs.mujoco.reacher import ReacherEnv


def experiment(variant):
    # env = MultitaskReacherEnv()
    env = gym_env("Reacher-v1")
    env = convert_to_tf_env(env)
    env = normalize(env)
    es = OUStrategy(action_space=env.action_space)
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    exploration_policy = ZeroPolicy(
        int(env.action_space.flat_dim),
    )
    policy = FeedForwardPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
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
    exp_prefix = "7-8-dev-state-distance-ddpg-baseline"

    # n_seeds = 10
    # mode = "ec2"
    # exp_prefix = "7-8-state-distance-ddpg-baseline"

    num_steps_per_iteration = 100
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
            discount=0.99,
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
