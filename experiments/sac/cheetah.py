"""
Run PyTorch Soft Actor Critic on HalfCheetahEnv.
"""
import random

import numpy as np
import gym

import railrl.torch.pytorch_util as ptu
from railrl.envs.wrappers import normalize_box
from railrl.launchers.launcher_util import run_experiment
from railrl.sac.policies import TanhGaussianPolicy
from railrl.sac.sac import SoftActorCritic
from railrl.torch.networks import FlattenMlp
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize
from rllab.envs.gym_env import GymEnv


def experiment(variant):
    # env = normalize(GymEnv(
    #     'HalfCheetah-v1',
    #     force_reset=True,
    #     record_video=False,
    #     record_log=False,
    # ))
    env = normalize_box(gym.make('HalfCheetah-v1'))

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
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
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=64,
            max_path_length=1000,
            discount=0.99,

            soft_target_tau=0.01,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        net_size=300,
    )
    for _ in range(1):
        seed = random.randint(0, 999999)
        run_experiment(
            experiment,
            seed=seed,
            variant=variant,
            exp_prefix="sac-half-cheetah-check",
            mode='ec2',
            # exp_prefix="dev-sac-half-cheetah",
            # mode='local',
            # use_gpu=True,
        )
