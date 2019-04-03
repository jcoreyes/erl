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
from railrl.torch.sac.twin_sac import TwinSACTrainer
import railrl.misc.hyperparameter as hyp
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from railrl.core.ray_experiment import RayExperiment
import ray
import ray.tune as tune

ENV_PARAMS = {
    'half-cheetah': {  # 6 DoF
        'env_class': HalfCheetahEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 1000,
        'train_policy_with_reparameterization': True,
    },
    'inv-double-pendulum': {  # 2 DoF
        'env_class': InvertedDoublePendulumEnv,
        'num_epochs': 100,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'train_policy_with_reparameterization': True,
    },
    'pendulum': {  # 2 DoF
        'env_class': PendulumEnv,
        'num_epochs': 20,
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

@tune.function
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
    vf = FlattenMlp(
        input_size=obs_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_vf = FlattenMlp(
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
    trainer = TwinSACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        target_vf=target_vf,
        discount=variant['discount'],
        soft_target_tau=variant['soft_target_tau'],
        policy_update_period=variant['policy_update_period'],
        target_update_period=variant['target_update_period'],
        train_policy_with_reparameterization=variant['train_policy_with_reparameterization'],
        policy_lr=variant['policy_lr'],
        qf_lr=variant['qf_lr'],
        vf_lr=variant['vf_lr'],
        reward_scale=1,
        use_automatic_entropy_tuning=True,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        data_buffer=replay_buffer,
        max_path_length=variant['max_path_length'],
        batch_size=variant['batch_size'],
        num_epochs=variant['num_epochs'],
        num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
        num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
        num_trains_per_train_loop=variant['num_trains_per_train_loop'],
        min_num_steps_before_training=variant['min_num_steps_before_training'],
    )
    return algorithm

def experiment(variant):
    ray.init(local_mode=True)
    # ray.init()
    import railrl.torch.pytorch_util as ptu

    exp = tune.Experiment(
        name="debug_test",
        run=RayExperiment,
        # num_samples=20,
        stop={"global_done": True},
        config={
            'algo_variant': variant,
            'init_algo_function': run_experiment_func,
            'use_gpu': ptu._use_gpu,
            'test': tune.grid_search([16, 64, 256]),
        },
        resources_per_trial={
            "cpu": 1,
            "gpu": 0.5,
        },
        checkpoint_freq=1,
    )
    tune.run(exp, resume=False)


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
        layer_size=256,
        algorithm="Twin-SAC",
        version="normal",
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    # n_seeds = 5
    # mode = 'sss'
    # exp_prefix = 'reference-twin-sac-post-mod-ref-min-num-steps'

    search_space = {
        'env': [
            # 'half-cheetah',
            # 'inv-double-pendulum',
            'pendulum',
            # 'ant',
            # 'walker',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                use_gpu=True,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
                time_in_mins=2 * 24 * 60,  # if you use mode=sss
            )
