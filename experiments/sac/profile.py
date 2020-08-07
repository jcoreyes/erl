"""
Profile SAC
"""

import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.multigoal import MultiGoalEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import ConcatMlp
import torch


def experiment(variant):
    env = NormalizedBoxEnv(MultiGoalEnv(
        actuation_cost_coeff=10,
        distance_cost_coeff=1,
        goal_reward=10,
    ))

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    qf = ConcatMlp(
        hidden_sizes=[100, 100],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = ConcatMlp(
        hidden_sizes=[100, 100],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[100, 100],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    algorithm = SoftActorCritic(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    with torch.autograd.profiler.profile() as prof:
        algorithm.train()
    prof.export_chrome_trace("tmp-torch-chrome-trace.prof")


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=10,
            num_steps_per_epoch=1000,
            num_steps_per_eval=300,
            batch_size=64,
            max_path_length=30,
            reward_scale=0.3,
            discount=0.99,
            soft_target_tau=0.001,
        ),
    )
    setup_logger("11-24-profile")
    experiment(variant)
