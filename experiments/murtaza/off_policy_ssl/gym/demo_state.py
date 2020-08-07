from rlkit.demos.td3_bc import TD3BCTrainer
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epislon import GaussianAndEpislonStrategy
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
import rlkit.misc.hyperparameter as hyp
from rlkit.torch.td3.td3 import TD3
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)

from gym.envs.mujoco import (
    HalfCheetahEnv,
    AntEnv,
    Walker2dEnv,
    HopperEnv,
)

from rlkit.launchers.launcher_util import run_experiment

ENV_PARAMS = {
    'half-cheetah': {  # 6 DoF
        'env_class': HalfCheetahEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 1000,
        'demo_path':"demos/hc_action_noise_1000.npy",
    },
    'hopper': {  # 6 DoF
        'env_class': HopperEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 1000,
        'demo_path':"demos/hopper_action_noise_1000.npy",
    },
    'ant': {  # 6 DoF
        'env_class': AntEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 3000,
        'demo_path':"demos/ant_action_noise_1000.npy",
    },
    'walker': {  # 6 DoF
        'env_class': Walker2dEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 3000,
        'demo_path':"demos/walker_action_noise_1000.npy",
    },
}

def experiment(variant):
    env_params = ENV_PARAMS[variant['env']]
    variant.update(env_params)

    expl_env = NormalizedBoxEnv(variant['env_class']())
    eval_env = NormalizedBoxEnv(variant['env_class']())
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    es = GaussianAndEpislonStrategy(
        action_space=expl_env.action_space,
        max_sigma=.2,
        min_sigma=.2,  # constant sigma
        epsilon=.3,
    )
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
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[M, M],
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[M, M],
    )
    expl_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        policy,
    )

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    demo_train_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    demo_test_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    if variant.get('td3_bc', True):
        td3_trainer = TD3BCTrainer(
            env=expl_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            target_policy=target_policy,
            demo_path=env_params['demo_path'],
            **variant['td3_bc_trainer_kwargs']
        )
    else:
        td3_trainer = TD3(
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            target_policy=target_policy,
            **variant['td3_trainer_kwargs']
        )
    if variant['collection_mode'] == 'online':
        expl_path_collector = MdpStepCollector(
            expl_env,
            expl_policy,
        )
        algorithm = TorchOnlineRLAlgorithm(
            trainer=td3_trainer,
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
            expl_policy,
        )
        algorithm = TorchBatchRLAlgorithm(
            trainer=td3_trainer,
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
    algorithm.to(ptu.device)
    if variant.get('load_demos', False):
        td3_trainer.load_demos()
    if variant.get('pretrain_policy', False):
        td3_trainer.pretrain_policy_with_bc()
    if variant.get('pretrain_rl', False):
        td3_trainer.pretrain_q_with_bc_data()
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
        algorithm="TD3 BC",
        version="normal",
        collection_mode='batch',
        td3_bc=True,
        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,
        td3_bc_trainer_kwargs=dict(
            discount=0.99,
            demo_off_policy_path=None,
            bc_num_pretrain_steps=10,
            q_num_pretrain_steps=20,
            rl_weight=1.0,
            bc_weight=0,
            reward_scale=1.0,
            weight_decay=0.0001,
            target_update_period=2,
            policy_update_period=2,
            beta=0.0001,
            max_path_length=1000,
            goal_conditioned=False,
            use_demo_awr=False,
        ),
    )

    search_space = {
        'td3_bc_trainer_kwargs.rl_weight':[0, 1.0],
        'td3_bc_trainer_kwargs.bc_weight':[0, 1.0],
        'td3_bc_trainer_kwargs.use_demo_awr':[True, False],
        'td3_bc_trainer_kwargs.awr_policy_update':[True, False],
        'env': [
            'half-cheetah',
            'ant',
            'walker',
            'hopper',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_name = 'test'

    # n_seeds = 2
    # mode = 'ec2'
    # exp_name = 'gym_demos_exps_v2'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        # if variant['td3_bc_trainer_kwargs']['bc_weight'] == 0 and variant['td3_bc_trainer_kwargs']['demo_beta'] != 1:
        #     continue
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                num_exps_per_instance=2,
                skip_wait=False,
                gcp_kwargs=dict(
                    preemptible=False,
                )
            )
