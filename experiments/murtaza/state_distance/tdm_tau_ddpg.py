import random
import rlkit.misc.hyperparameter as hyp
from rlkit.data_management.her_replay_buffer import HerReplayBuffer
from rlkit.envs.multitask.ant_env import GoalXYPosAnt
from rlkit.envs.multitask.half_cheetah import GoalXVelHalfCheetah
from rlkit.envs.multitask.her_half_cheetah import HalfCheetah
from rlkit.envs.multitask.pusher2d import MultitaskPusher2DEnv
from rlkit.envs.multitask.pusher3d import MultitaskPusher3DEnv
from rlkit.envs.multitask.reacher_7dof import Reacher7DofXyzGoalState
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.state_distance.experimental_tdm_networks import StructuredQF, \
    OneHotTauQF, BinaryStringTauQF, TauVectorQF, TauVectorSeparateFirstLayerQF, \
    StandardTdmPolicy, OneHotTauTdmPolicy, BinaryTauTdmPolicy, \
    TauVectorTdmPolicy, TauVectorSeparateFirstLayerTdmPolicy
from rlkit.state_distance.tdm_networks import *
from rlkit.state_distance.tdm_ddpg import TdmDdpg

import rlkit.torch.pytorch_util as ptu

def experiment(variant):
    env = NormalizedBoxEnv(variant['env_class']())

    obs_dim = int(np.prod(env.observation_space.low.shape))
    action_dim = int(np.prod(env.action_space.low.shape))
    vectorized = variant['algo_params']['tdm_kwargs']['vectorized']
    qf_class = variant['qf_class']
    policy_class = variant['policy_class']
    es_class = variant['es_class']
    # obs_normalizer = TorchFixedNormalizer(obs_dim)
    # goal_normalizer = TorchFixedNormalizer(env.goal_dim)
    # action_normalizer = TorchFixedNormalizer(action_dim)
    # tau_normalizer = TorchFixedNormalizer(1)

    es_params = dict(
        action_space=env.action_space,
        **variant['es_params']
    )
    es = es_class(**es_params)
    # qf = MakeNormalizedTDMQF(qf_class(
    #     observation_dim=obs_dim,
    #     action_dim=action_dim,
    #     goal_dim =env.goal_dim,
    #     output_size=env.goal_dim if vectorized else 1,
    #     **variant['qf_params']
    #     ),
    #     env,
    #     obs_normalizer=obs_normalizer,
    #     goal_normalizer=goal_normalizer,
    #     action_normalizer=action_normalizer,
    #     tau_normalizer=tau_normalizer,
    # )
    # policy = MakeNormalizedTDMPolicy(policy_class(
    #         obs_dim=obs_dim,
    #         action_dim=action_dim,
    #         goal_dim=env.goal_dim,
    #         **variant['policy_params']
    #     ),
    #     env,
    #     obs_normalizer=obs_normalizer,
    #     goal_normalizer=goal_normalizer,
    #     tau_normalizer=tau_normalizer,
    # )
    qf = qf_class(
        observation_dim=obs_dim,
        action_dim=action_dim,
        goal_dim =env.goal_dim,
        output_size=env.goal_dim if vectorized else 1,
        **variant['qf_params']
        )
    policy = policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            goal_dim=env.goal_dim,
            **variant['policy_params']
        )
    replay_buffer = HerReplayBuffer(
        env=env,
        **variant['her_replay_buffer_params']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = TdmDdpg(
        env=env,
        policy=policy,
        qf=qf,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "ddpg"

    num_epochs = 100
    num_steps_per_epoch = 1000
    num_steps_per_eval = 1000
    max_path_length = 100
    max_tau = max_path_length-1
    # noinspection PyTypeChecker
    versions = [
        # (StructuredQF, StructuredQF, StandardTdmPolicy, '_standard'),
        (OneHotTauQF, OneHotTauQF, OneHotTauTdmPolicy, '_one_hot_tau'),
        # (BinaryStringTauQF, BinaryStringTauQF, BinaryTauTdmPolicy, '_binary_string_tau'),
    ]
    variant = dict(
        algo_params=dict(
            base_kwargs=dict(
                num_epochs=num_epochs,
                num_steps_per_epoch=num_steps_per_epoch,
                num_steps_per_eval=num_steps_per_eval,
                max_path_length=max_path_length,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=1,
            ),
            tdm_kwargs=dict(
                sample_rollout_goals_from='environment',
                sample_train_goals_from='her',
                vectorized=True,
                cycle_taus_for_rollout=True,
                max_tau=10,
            ),
            ddpg_kwargs=dict(
                tau=0.001,
                qf_learning_rate=1e-3,
                policy_learning_rate=1e-4,
            ),
        ),
        her_replay_buffer_params=dict(
            max_size=int(2E5),
            num_goals_to_sample=4,
        ),
        qf_params=dict(
            max_tau=max_tau,
            hidden_sizes=[100, 100],
        ),
        vf_params=dict(
            max_tau=max_tau,
            hidden_sizes=[100, 100],
        ),
        policy_params=dict(
            max_tau=max_tau,
            hidden_sizes=[100, 100],
        ),
        es_params=dict(
            min_sigma=0.25,
            max_sigma=0.25,
        ),
        es_class=OUStrategy,
    )
    search_space = {
        'env_class': [
            MultitaskPusher3DEnv,
            # GoalXVelHalfCheetah,
        ],
        'algo_params.base_kwargs.reward_scale': [
            1,
            10,
            100,
            1000,
            10000,
        ],
        'algo_params.tdm_kwargs.vectorized': [
            True,
        ],
        'algo_params.tdm_kwargs.sample_rollout_goals_from': [
            'environment',
        ],
        'algo_params.base_kwargs.num_updates_per_env_step':[
            # 1,
            # 5,
            # 10,
            15,
            20,
            25,
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for version in versions:
        qf_class = version[0]
        vf_class=version[1]
        policy_class=version[2]
        exp = version[3]
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            variant['qf_class']=qf_class
            variant['vf_class']=vf_class
            variant['policy_class']=policy_class
            experiment_prefix = exp_prefix
            if variant['env_class']==GoalXVelHalfCheetah:
                experiment_prefix += '_HalfCheetah'
            elif variant['env_class'] == MultitaskPusher3DEnv:
                experiment_prefix += '_Pusher3D'
            for i in range(n_seeds):
                seed = random.randint(0, 10000)
                run_experiment(
                    experiment,
                    seed=seed,
                    variant=variant,
                    exp_id=exp_id,
                    exp_prefix=experiment_prefix+exp,
                    mode=mode,
                )
