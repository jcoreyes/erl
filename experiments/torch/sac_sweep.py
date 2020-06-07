import numpy as np
from gym.envs.mujoco import (
    HalfCheetahEnv,
    AntEnv,
    HopperEnv,
    Walker2dEnv,
)

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.torch.sac.sac import SoftActorCritic
from railrl.torch.networks import ConcatMlp


def experiment(variant):
    env = variant['env_class']()
    env = NormalizedBoxEnv(env)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    qf = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    vf = ConcatMlp(
        input_size=obs_dim,
        output_size=1,
        **variant['vf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    algorithm = SoftActorCritic(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=200,
            num_steps_per_epoch=5000,
            num_steps_per_eval=10000,
            max_path_length=1000,
            min_num_steps_before_training=10000,
            batch_size=128,
            discount=0.99,

            save_replay_buffer=False,
            replay_buffer_size=int(1E6),
            train_policy_with_reparameterization=True,

            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        vf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        algorithm='SAC',
        version='SAC',
        env_class=HalfCheetahEnv,
    )
    search_space = {
        'env_class': [
            HalfCheetahEnv,
            AntEnv,
            HopperEnv,
            Walker2dEnv,
        ],
        'algo_kwargs.reward_scale': [0.1, 1, 10],
        # 'algo_kwargs.num_updates_per_env_step': [1, 5],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(2):
            run_experiment(
                experiment,
                # exp_prefix="dev-sac-sweep",
                exp_prefix="sac-sweep-try-reparameterization",
                mode='ec2',
                exp_id=exp_id,
                variant=variant,
                use_gpu=False,
            )
