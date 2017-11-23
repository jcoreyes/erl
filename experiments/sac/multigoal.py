"""
Run PyTorch Soft Actor Critic on Multigoal Env.
"""
import random

import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.envs.multigoal import MultiGoalEnv
from railrl.envs.wrappers import normalize_box
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.plotter import QFPolicyPlotter
from railrl.sac.policies import TanhGaussianPolicy
from railrl.sac.sac import SoftActorCritic
from railrl.torch.networks import FlattenMlp
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv


def experiment(variant):
    env = normalize_box(MultiGoalEnv(
        actuation_cost_coeff=10,
        distance_cost_coeff=1,
        goal_reward=10,
    ))

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
    plotter = QFPolicyPlotter(
        qf=qf,
        policy=policy,
        obs_lst=np.array([[-2.5, 0.0],
                          [0.0, 0.0],
                          [2.5, 2.5]]),
        default_action=[np.nan, np.nan],
        n_samples=100
    )
    algorithm = SoftActorCritic(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        plotter=plotter,
        render_eval_paths=True,
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
