"""
Test twin sac against various environments.
"""
from rlkit.envs.erl import (
    HalfCheetahEnv,
    AntEnv,
    Walker2dEnv,
    InvertedDoublePendulumEnv,
    HopperEnv,
    HumanoidEnv,
    SwimmerEnv,
)
from gym.envs.classic_control import PendulumEnv

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
import rlkit.misc.hyperparameter as hyp
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithmModEnv

ENV_PARAMS = {
    'half-cheetah': {  # 6 DoF
        'env_class': HalfCheetahEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 1000,
    },
    'hopper': {  # 6 DoF
        'env_class': HopperEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 1000,
    },
    'humanoid': {  # 6 DoF
        'env_class': HumanoidEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 3000,
    },
    'inv-double-pendulum': {  # 2 DoF
        'env_class': InvertedDoublePendulumEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 100,
    },
    'pendulum': {  # 2 DoF
        'env_class': PendulumEnv,
        'num_expl_steps_per_train_loop': 200,
        'max_path_length': 200,
        'num_epochs': 200,
        'min_num_steps_before_training': 2000,
        'target_update_period': 200,
    },
    'ant': {  # 6 DoF
        'env_class': AntEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 3000,
    },
    'walker': {  # 6 DoF
        'env_class': Walker2dEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 3000,
    },
    'swimmer': {  # 6 DoF
        'env_class': SwimmerEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 2000,
    },
}


def experiment(variant):
    env_params = ENV_PARAMS[variant['env']]
    env_mod_params = variant['env_mod']
    variant.update(env_params)

    expl_env = NormalizedBoxEnv(variant['env_class'](env_mod_params))
    eval_env = NormalizedBoxEnv(variant['env_class']({}))
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
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
        **variant['trainer_kwargs']
    )
    if variant['collection_mode'] == 'online':
        expl_path_collector = MdpStepCollector(
            expl_env,
            policy,
        )
        algorithm = TorchOnlineRLAlgorithm(
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
    else:
        expl_path_collector = MdpPathCollector(
            expl_env,
            policy,
        )
        algorithm = TorchBatchRLAlgorithmModEnv(
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
            mod_env_epoch_schedule=variant['mod_env_epoch_schedule'],
            env_class=variant['env_class'],
            env_mod_params=variant['env_mod']

        )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        num_epochs=3000,
        num_eval_steps_per_epoch=5000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
        batch_size=256,
        replay_buffer_size=int(1E6),
        layer_size=256,
        algorithm="SAC",
        version="normal",
        collection_mode='batch',
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        mod_env_epoch_schedule=0.5,
    )
    num_configurations = 200
    n_seeds = 1
    mode = 'ec2'
    exp_name = 'dev'

    # n_seeds = 5
    # mode = 'sss'
    # exp_name = 'rlkit-half-cheetah-online'
    env_ids = [
            'half-cheetah',
             'inv-double-pendulum',
            # 'pendulum',
             'ant',
             'walker',
             'hopper',
             'humanoid',
             'swimmer',
        ]

    search_space = {
        'env': [
            'half-cheetah',
             'inv-double-pendulum',
            # 'pendulum',
             'ant',
             'walker',
             'hopper',
             'humanoid',
             'swimmer',
        ],
        'env_mod': {
            'gravity': [0.9, 1],
            'friction': [0.9, 1],
            'ctrlrange': [0.9, 1],
            'gear': [0.9, 1],

        }
    }
    # hyperparameters = [
    #     hyp.LinearFloatParam('foo', 0, 1),
    #     hyp.LogFloatParam('bar', 1e-5, 1e2),
    # ]
    # sweeper = hyp.RandomHyperparameterSweeper(
    #     hyperparameters,
    #     default_kwargs=variant,
    # )

    # mods = dict(gravity=0.9,
    #             friction=0.9,
    #             ctrlrange=0.9,
    #             gear=0.9
    #             )
    #
    # sweeper = hyp.DeterministicHyperparameterSweeper(
    #     search_space, default_parameters=variant,
    # )
    hyperparameters = [
        hyp.LinearFloatParam('env_mod.gravity', 0.5, 1.2),
        hyp.LinearFloatParam('env_mod.friction', 0.5, 1.2),
        hyp.LinearFloatParam('env_mod.ctrlrange', 0.5, 1.2),
        hyp.LinearFloatParam('env_mod.gear', 0.5, 1.2),
    ]
    sweeper = hyp.RandomHyperparameterSweeper(
        hyperparameters,
        default_kwargs=variant,
    )
    for exp_id in range(num_configurations):
        variant = sweeper.generate_random_hyperparameters()
        for env_id in env_ids:
            variant['env'] = env_id
            for _ in range(n_seeds):
                run_experiment(
                    experiment,
                    unpack_variant=False,
                    exp_name=exp_name,
                    mode=mode,
                    variant=variant,
                    time_in_mins=int(2.8 * 24 * 60),  # if you use mode=sss
                    region='us-east-2',
                )

    # for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
    #     for _ in range(n_seeds):
    #         variant['exp_id'] = exp_id
    #         run_experiment(
    #             experiment,
    #             unpack_variant=False,
    #             exp_name=exp_name,
    #             mode=mode,
    #             variant=variant,
    #             time_in_mins=int(2.8 * 24 * 60),  # if you use mode=sss
    #             region='us-east-2',
    #         )
    #         break
