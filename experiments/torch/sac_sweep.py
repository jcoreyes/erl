import numpy as np
import torch.optim as optim
from gym.envs.mujoco import HalfCheetahEnv

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.envs.pygame.point2d import Point2DEnv
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.torch.sac.sac import SoftActorCritic
from railrl.torch.networks import FlattenMlp


def experiment(variant):
    env = NormalizedBoxEnv(variant['env_class']())

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
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=10,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,

            save_replay_buffer=True,
            replay_buffer_size=15000,

            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        net_size=300,
        algorithm='SAC',
        version='SAC',
        env_class=HalfCheetahEnv,
    )
    search_space = {
        'env_class': [
            # HalfCheetahEnv,
            Point2DEnv,
        ],
        'algo_kwargs.reward_scale': [
            1,
        ],
        'algo_kwargs.optimizer_class': [
            optim.Adam,
        ],
        'algo_kwargs.soft_target_tau': [
            1e-3,
        ],
        'algo_kwargs.num_updates_per_env_step': [
            1,
        ],
        'algo_kwargs.policy_mean_reg_weight': [
            1e-3
        ],
        'algo_kwargs.policy_std_reg_weight': [
            1e-3
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(1):
            run_experiment(
                experiment,
                # exp_prefix="dev-sac-sweep",
                exp_prefix="sac-point2d-short",
                # mode='ec2',
                exp_id=exp_id,
                variant=variant,
                use_gpu=False,
            )