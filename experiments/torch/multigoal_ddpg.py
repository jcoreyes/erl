import random
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.multigoal import MultiGoalEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.visualization.plotter import QFPolicyPlotter
from rlkit.torch.networks import FeedForwardQFunction, FeedForwardPolicy
from rlkit.torch.ddpg.ddpg import DDPG


def experiment(variant):
    env = NormalizedBoxEnv(MultiGoalEnv(
        actuation_cost_coeff=10,
        distance_cost_coeff=1,
        goal_reward=10,
    ))

    es = OUStrategy(action_space=env.action_space)
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        100,
        100,
    )
    policy = FeedForwardPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        100,
        100,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    plotter = QFPolicyPlotter(
        qf=qf,
        # policy=policy,
        policy=exploration_policy,
        obs_lst=np.array([[-2.5, 0.0],
                          [0.0, 0.0],
                          [2.5, 2.5]]),
        default_action=[np.nan, np.nan],
        n_samples=100
    )
    algorithm = DDPG(
        env,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        render_eval_paths=True,
        plotter=plotter,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=300,
            batch_size=64,
            max_path_length=30,
            reward_scale=0.3,
            discount=0.99,
            tau=0.001,
        ),
    )
    for _ in range(1):
        seed = random.randint(0, 999999)
        run_experiment(
            experiment,
            seed=seed,
            variant=variant,
            # exp_prefix="sac-half-cheetah-with-action-hack",
            # mode='ec2',
            # use_gpu=False,
            exp_prefix="dev-sac-half-cheetah",
            mode='local',
            use_gpu=True,
        )
