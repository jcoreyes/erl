import random
import numpy as np

from railrl.data_management.her_replay_buffer import HerReplayBuffer
import railrl.torch.pytorch_util as ptu
from railrl.envs.multitask.sawyer_env import MultiTaskSawyerEnv
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.state_distance.tdm_networks import StructuredQF
from railrl.state_distance.tdm_ddpg import TdmDdpg
from railrl.torch.modules import HuberLoss


def experiment(variant):
    env = MultiTaskSawyerEnv(**variant['env_params'])
    obs_dim = int(np.prod(env.observation_space.low.shape))
    action_dim = int(np.prod(env.action_space.low.shape))
    vectorized = variant['algo_params']['tdm_kwargs']['vectorized']
    qf = StructuredQF(
        observation_dim=obs_dim,
        action_dim=action_dim,
        goal_dim=env.goal_dim,
        output_size=env.goal_dim if vectorized else 1,
        **variant['qf_params']
    )
    policy = FeedForwardPolicy(
        obs_dim=obs_dim + env.goal_dim + 1,
        action_dim=action_dim,
        **variant['policy_params']
    )
    es = OUStrategy(
        action_space=env.action_space,
        theta=0.1,
        max_sigma=0.1,
        min_sigma=0.1,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = HerReplayBuffer(
        env=env,
        **variant['her_replay_buffer_params']
    )
    qf_criterion = variant['qf_criterion_class'](
        **variant['qf_criterion_params']
    )
    algo_params = variant['algo_params']
    algo_params['ddpg_kwargs']['qf_criterion'] = qf_criterion
    algorithm = TdmDdpg(
        env,
        qf=qf,
        replay_buffer=replay_buffer,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

experiments=[
    'joint_angle|fixed_angle',
    'joint_angle|varying_angle',
    'end_effector_position|fixed_ee',
    'end_effector_position|varying_ee',
    'end_effector_position_orientation|fixed_ee',
    'end_effector_position_orientation|varying_ee'
]

if __name__ == "__main__":
    n_seeds = 3
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            base_kwargs=dict(
                num_epochs=60,
                num_steps_per_epoch=1000,
                num_steps_per_eval=500,
                num_updates_per_env_step=3,
                batch_size=64,
                max_path_length=100,
                discount=1,
                collection_mode='online'
            ),
            tdm_kwargs=dict(
                sample_rollout_goals_from='environment',
                sample_train_goals_from='her',
                vectorized=False,
            ),
            ddpg_kwargs=dict(
                tau=0.001,
                qf_learning_rate=1e-3,
                policy_learning_rate=1e-4,
            ),
        ),
        sampler_es_class=OUStrategy,
        sampler_es_params=dict(
            theta=0.1,
            max_sigma=0.25,
            min_sigma=0.25,
        ),
        her_replay_buffer_params=dict(
            max_size=200000,
            num_goals_to_sample=4,
        ),
        env_params={
            'arm_name': 'right',
            'safety_box': True,
            'loss': 'lorentz',
            'huber_delta': 0.01,
            'safety_force_magnitude': 7,
            'temp': 15,
            'remove_action': False,
            'experiment': experiments[4],
            'reward_magnitude': 10,
            'use_safety_checks': False,
            'task': 'lego',
        },
        qf_params=dict(
            hidden_sizes=[300, 300],
        ),
        policy_params=dict(
            fc1_size=300,
            fc2_size=300,
        ),
        qf_criterion_class=HuberLoss,
        qf_criterion_params=dict(),
    )
    run_experiment(
        experiment,
        seed=random.randint(0, 10000),
        exp_prefix="TDM-Final-sawyer-lego-block-stacking-lorentz",
        mode="local",
        variant=variant,
        exp_id=0,
        use_gpu=True,
        snapshot_mode="last",
    )
    variant = dict(
        algo_params=dict(
            base_kwargs=dict(
                num_epochs=60,
                num_steps_per_epoch=1000,
                num_steps_per_eval=500,
                num_updates_per_env_step=3,
                batch_size=64,
                max_path_length=100,
                discount=1,
            ),
            tdm_kwargs=dict(
                sample_rollout_goals_from='environment',
                sample_train_goals_from='her',
                vectorized=True,
            ),
            ddpg_kwargs=dict(
                tau=0.001,
                qf_learning_rate=1e-3,
                policy_learning_rate=1e-4,
            ),
        ),
        sampler_es_class=OUStrategy,
        sampler_es_params=dict(
            theta=0.1,
            max_sigma=0.25,
            min_sigma=0.25,
        ),
        her_replay_buffer_params=dict(
            max_size=200000,
            num_goals_to_sample=4,
        ),
        env_params={
            'arm_name': 'right',
            'safety_box': True,
            'loss': 'norm',
            'huber_delta': 10,
            'safety_force_magnitude': 7,
            'temp': 15,
            'remove_action': False,
            'experiment': experiments[4],
            'reward_magnitude': 10,
            'use_safety_checks': False,
            'task': 'lego',
        },
        qf_params=dict(
            hidden_sizes=[300, 300],
        ),
        policy_params=dict(
            fc1_size=300,
            fc2_size=300,
        ),
        qf_criterion_class=HuberLoss,
        qf_criterion_params=dict(),
    )
    run_experiment(
        experiment,
        seed=random.randint(0, 10000),
        exp_prefix="TDM-Final-sawyer-lego-block-stacking-norm-vectorized",
        mode="local",
        variant=variant,
        exp_id=0,
        use_gpu=True,
        snapshot_mode="last",
    )
    variant = dict(
        algo_params=dict(
            base_kwargs=dict(
                num_epochs=60,
                num_steps_per_epoch=1000,
                num_steps_per_eval=500,
                num_updates_per_env_step=3,
                batch_size=64,
                max_path_length=100,
                discount=1,
            ),
            tdm_kwargs=dict(
                sample_rollout_goals_from='environment',
                sample_train_goals_from='her',
                vectorized=False,
            ),
            ddpg_kwargs=dict(
                tau=0.001,
                qf_learning_rate=1e-3,
                policy_learning_rate=1e-4,
            ),
        ),
        sampler_es_class=OUStrategy,
        sampler_es_params=dict(
            theta=0.1,
            max_sigma=0.25,
            min_sigma=0.25,
        ),
        her_replay_buffer_params=dict(
            max_size=200000,
            num_goals_to_sample=4,
        ),
        env_params={
            'arm_name': 'right',
            'safety_box': True,
            'loss': 'norm',
            'huber_delta': 10,
            'safety_force_magnitude': 7,
            'temp': 15,
            'remove_action': False,
            'experiment': experiments[4],
            'reward_magnitude': 10,
            'use_safety_checks': False,
            'task': 'lego',
        },
        qf_params=dict(
            hidden_sizes=[300, 300],
        ),
        policy_params=dict(
            fc1_size=300,
            fc2_size=300,
        ),
        qf_criterion_class=HuberLoss,
        qf_criterion_params=dict(),
    )
    run_experiment(
        experiment,
        seed=random.randint(0, 10000),
        exp_prefix="TDM-Final-sawyer-lego-block-stacking-norm",
        mode="local",
        variant=variant,
        exp_id=0,
        use_gpu=True,
        snapshot_mode="last",
    )

#different variants to run:
#implement different types of losses for batch computation
'''
TDM online lorentz not vectorized
TDM parallel lorentz not vectorized
TDM online lorentz vectorized
TDM parallel lorentz vectorized
'''
