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
from railrl.sac.expected_sac import ExpectedSAC
from railrl.sac.policies import TanhGaussianPolicy
from railrl.sac.sac import SoftActorCritic
from railrl.sac.value_functions import ExpectableQF
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

    qf = ExpectableQF(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=100,
    )
    # qf = FlattenMlp(
    #     hidden_sizes=[100],
    #     input_size=obs_dim + action_dim,
    #     output_size=1,
    # )
    vf = FlattenMlp(
        hidden_sizes=[100],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[100],
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
    algorithm = ExpectedSAC(
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
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=300,
            batch_size=64,
            max_path_length=30,
            reward_scale=0.3,
            discount=0.99,
            soft_target_tau=0.001,
            naive_expectation=True,
        ),
    )
    for _ in range(1):
        seed = random.randint(0, 999999)
        run_experiment(
            experiment,
            seed=seed,
            variant=variant,
            exp_prefix="dev-sac-multigoal",
            mode='local',
            use_gpu=True,
        )
