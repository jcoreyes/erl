"""
Run PyTorch Soft Actor Critic on HalfCheetahEnv.
"""
import random

import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.launchers.launcher_util import run_experiment
from railrl.sac.policies import TanhGaussianPolicy
from railrl.sac.sac import SoftActorCritic
from railrl.torch.networks import FlattenMlp
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv


def experiment(variant):
    env = HalfCheetahEnv()

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    qf = FlattenMlp(
        hidden_sizes=[100, 100],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
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
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=100,
            num_steps_per_eval=100,
            batch_size=32,
            max_path_length=10,
            discount=0.99,
            soft_target_tau=0.001,
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
