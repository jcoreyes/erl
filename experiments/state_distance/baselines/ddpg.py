import random

from railrl.envs.env_utils import gym_env
from railrl.envs.wrappers import normalize_box
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG


def experiment(variant):
    env = gym_env("Reacher-v1")
    env = normalize_box(
        env,
        **variant['normalize_params']
    )
    es = OUStrategy(action_space=env.action_space)
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    policy = FeedForwardPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = DDPG(
        env,
        qf,
        policy,
        exploration_policy,
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-state-distance-ddpg-baseline"

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "ddpg-reacher-nupo-sweep-old-net-size-no-normalization"

    num_steps_per_iteration = 900
    H = 300
    num_iterations = 100
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=num_iterations,
            num_steps_per_epoch=num_steps_per_iteration,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=H,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            number_of_gradient_steps=1,
        ),
        version="DDPG",
        normalize_params=dict(
            obs_mean=None,
            obs_std=None,
        ),
    )
    for i, nupo in enumerate([1, 10, 50]):
        variant['algo_params']['number_of_gradient_steps'] = nupo
        for _ in range(n_seeds):
            seed = random.randint(0, 999999)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                exp_id=i,
                seed=seed,
                mode=mode,
                variant=variant,
                use_gpu=False,
            )
