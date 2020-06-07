""" Run PyTorch Soft Actor Critic on Multigoal Env.
"""
import random

import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.sac.expected_sac import ExpectedSAC
from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.torch.networks import ConcatMlp
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
import railrl.misc.hyperparameter as hyp


def experiment(variant):
    # env = NormalizedBoxEnv(MultiGoalEnv(
    #     actuation_cost_coeff=10,
    #     distance_cost_coeff=1,
    #     goal_reward=10,
    # ))
    env = NormalizedBoxEnv(HalfCheetahEnv())

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    # qf = ExpectableQF(
        # obs_dim=obs_dim,
        # action_dim=action_dim,
        # hidden_size=100,
    # )
    net_size = variant['net_size']
    qf = ConcatMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = ConcatMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    # TODO(vitchyr): just creating the plotter crashes EC2
    # plotter = QFPolicyPlotter(
        # qf=qf,
        # policy=policy,
        # obs_lst=np.array([[-2.5, 0.0],
                          # [0.0, 0.0],
                          # [2.5, 2.5]]),
        # default_action=[np.nan, np.nan],
        # n_samples=100
    # )
    algorithm = ExpectedSAC(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        # plotter=plotter,
        # render_eval_paths=True,
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
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=999,
            discount=0.99,

            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,

            expected_qf_estim_strategy='sample',
            expected_log_pi_estim_strategy='sample',
        ),
        net_size=300,
        version="original-normal-qf",
    )
    search_space = {
        'algo_params.expected_qf_estim_strategy': [
            'mean_action',
            'sample',
        ],
        'algo_params.expected_log_pi_estim_strategy': [
            'mean_action',
            'sample',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(3):
            seed = random.randint(0, 999999)
            run_experiment(
                experiment,
                exp_id=exp_id,
                seed=seed,
                variant=variant,
                exp_prefix="expected-sac-cheetah-sweep-2",
                mode='ec2',
                use_gpu=False,
                # exp_prefix="dev-expected-sac-cheetah-sweep",
                # mode='local',
            )
