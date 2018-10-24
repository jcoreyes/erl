import random

import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.networks import FlattenMlp

from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.torch.sac.sac import SoftActorCritic

import railrl.misc.hyperparameter as hyp
from railrl.envs.multitask.pusher import MultitaskPusherEnv


def experiment(variant):
    env = MultitaskPusherEnv()
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
    use_gpu=True
    variant = dict(
        algo_params=dict(
            num_epochs=150,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=100,
            discount=0.99,
            soft_target_tau=0,
            target_hard_update_period=5,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            render=True,
        ),
        net_size=100,
    )
    search_space = {
        'algo_params.reward_scale': [
            10,
            # 100,
            # 1000,
            # 10000,
        ],
        'algo_params.num_updates_per_env_step': [
            10,
            15,
            # 20,
            # 25,
        ],
        'algo_params.batch_size': [
            # 64,
            128,
            # 512,
            # 1024,
        ]

    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    n_seeds=1
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                seed=seed,
                variant=variant,
                exp_id=exp_id,
                exp_prefix='test',
                mode='local',
                use_gpu=use_gpu,
            )

