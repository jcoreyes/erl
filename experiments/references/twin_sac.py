"""
Test twin sac against various environments.
"""
from gym.envs.mujoco import (
    HalfCheetahEnv,
    AntEnv,
    Walker2dEnv,
    InvertedDoublePendulumEnv,
    HopperEnv,
    HumanoidEnv,
)
from gym.envs.classic_control import PendulumEnv

from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.launchers.launcher_util import run_experiment
import railrl.torch.pytorch_util as ptu
from railrl.torch.networks import FlattenMlp
from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.torch.sac.sac import SAC
import railrl.misc.hyperparameter as hyp


ENV_PARAMS = {
    'half-cheetah': {  # 6 DoF
        'env_class': HalfCheetahEnv,
        'num_epochs': 3000,
    },
    'hopper': {  # 6 DoF
        'env_class': HopperEnv,
        'num_epochs': 1000,
    },
    'humanoid': {  # 6 DoF
        'env_class': HumanoidEnv,
        'num_epochs': 3000,
    },
    'inv-double-pendulum': {  # 2 DoF
        'env_class': InvertedDoublePendulumEnv,
        'num_epochs': 100,
    },
    'pendulum': {  # 2 DoF
        'env_class': PendulumEnv,
        'num_epochs': 200,
        'num_steps_per_epoch': 200,
        'num_steps_per_eval': 200,
        'max_path_length': 200,
        'min_num_steps_before_training': 2000,
        'target_update_period': 200,
    },
    'ant': {  # 6 DoF
        'env_class': AntEnv,
        'num_epochs': 3000,
    },
    'walker': {  # 6 DoF
        'env_class': Walker2dEnv,
        'num_epochs': 3000,
    },
}


def experiment(variant):
    env_params = ENV_PARAMS[variant['env']]
    variant.update(env_params)

    env = NormalizedBoxEnv(variant['env_class']())
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    variant['algo_kwargs'] = dict(
        num_epochs=variant['num_epochs'],
        num_steps_per_epoch=variant['num_steps_per_epoch'],
        num_steps_per_eval=variant['num_steps_per_eval'],
        max_path_length=variant['max_path_length'],
        min_num_steps_before_training=variant['min_num_steps_before_training'],
        batch_size=variant['batch_size'],
        discount=variant['discount'],
        replay_buffer_size=variant['replay_buffer_size'],
        soft_target_tau=variant['soft_target_tau'],
        target_update_period=variant['target_update_period'],
        policy_lr=variant['policy_lr'],
        qf_lr=variant['qf_lr'],
        reward_scale=1,
        use_automatic_entropy_tuning=True,
    )

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    algorithm = SAC(
        env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        num_epochs=3000,
        num_steps_per_epoch=1000,
        num_steps_per_eval=1000,
        max_path_length=1000,
        min_num_steps_before_training=1000,
        batch_size=256,
        discount=0.99,
        replay_buffer_size=int(1E6),
        soft_target_tau=5e-3,
        target_update_period=1,
        policy_lr=3E-4,
        qf_lr=3E-4,
        layer_size=256, # [256, 512]
        algorithm="Twin-SAC",
        version="normal",
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 2
    mode = 'sss'
    exp_prefix = 'reference-new-tsac-pre-refactor'

    search_space = {
        'env': [
            # 'half-cheetah',
            'inv-double-pendulum',
            'pendulum',
            'ant',
            'walker',
            'hopper',
            'humanoid',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
                time_in_mins=2*24*60,  # if you use mode=sss
            )
