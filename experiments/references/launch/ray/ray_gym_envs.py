"""
Test twin sac against various environments.
"""
from gym.envs.mujoco import (
    HalfCheetahEnv,
    AntEnv,
    Walker2dEnv,
    InvertedDoublePendulumEnv,
)
from gym.envs.classic_control import PendulumEnv

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.samplers.data_collector import MdpPathCollector
from railrl.torch.networks import FlattenMlp
from railrl.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from railrl.torch.sac.sac import SACTrainer
import railrl.misc.hyperparameter as hyp
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import ray
import ray.tune as tune
from railrl.launchers.ray.launcher import launch_experiment

ENV_PARAMS = {
    'half-cheetah': {  # 6 DoF
        'env_class': HalfCheetahEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 200,
        'train_policy_with_reparameterization': True,
    },
    'inv-double-pendulum': {  # 2 DoF
        'env_class': InvertedDoublePendulumEnv,
        'num_epochs': 1000,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'train_policy_with_reparameterization': True,
    },
    'pendulum': {  # 2 DoF
        'env_class': PendulumEnv,
        'num_epochs': 1000,
        'num_expl_steps_per_train_loop': 200,
        'max_path_length': 200,
        'min_num_steps_before_training': 2000,
        'target_update_period': 200,
        'train_policy_with_reparameterization': False,
    },
    'ant': {  # 6 DoF
        'env_class': AntEnv,
        'num_epochs': 3000,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'train_policy_with_reparameterization': True,
    },
    'walker': {  # 6 DoF
        'env_class': Walker2dEnv,
        'num_epochs': 3000,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'train_policy_with_reparameterization': True,
    },
}

def run_experiment_func(variant):
    env_params = ENV_PARAMS[variant['env']]
    variant.update(env_params)

    expl_env = NormalizedBoxEnv(variant['env_class']())
    eval_env = NormalizedBoxEnv(variant['env_class']())
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

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
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )

    vf = FlattenMlp(
        input_size=obs_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        discount=variant['discount'],
        soft_target_tau=variant['soft_target_tau'],
        target_update_period=variant['target_update_period'],
        policy_lr=variant['policy_lr'],
        qf_lr=variant['qf_lr'],
        reward_scale=1,
        use_automatic_entropy_tuning=True,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=variant['max_path_length'],
        batch_size=variant['batch_size'],
        num_epochs=variant['num_epochs'],
        num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
        num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
        num_trains_per_train_loop=variant['num_trains_per_train_loop'],
        min_num_steps_before_training=variant['min_num_steps_before_training'],
    )
    return algorithm

if __name__ == "__main__":
    variant = dict(
        num_epochs=300,
        num_eval_steps_per_epoch=500,
        num_trains_per_train_loop=100,
        num_expl_steps_per_train_loop=100,
        min_num_steps_before_training=100,
        max_path_length=100,
        batch_size=256,
        discount=0.99,
        replay_buffer_size=int(1E6),
        soft_target_tau=1.0,
        policy_update_period=1,  # check
        target_update_period=100,  # check
        train_policy_with_reparameterization=False,
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
        env=tune.grid_search(['pendulum']),
        layer_size=256,
        algorithm="Twin-SAC",
        version="normal",
        # env='pendulum',
    )
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'aws-autoscaler-test-resume-kill-worker-new-settings-4'


    launch_experiment(
        mode=mode,
        use_gpu=False,
        local_launch_variant=dict(
            seeds=n_seeds,
            init_algo_functions_and_log_fnames=[(run_experiment_func, 'progress.csv')],
            exp_variant=variant,
            checkpoint_freq=20,
            exp_prefix=exp_prefix,
            resources_per_trial={
                'cpu': 2,
            }
        ),
        remote_launch_variant=dict(
            # head_instance_type='m1.xlarge',
            max_spot_price=.2,
        ),
        docker_variant=dict(),
    )

