import random

import rlkit.misc.hyperparameter as hyp
from rlkit.data_management.her_replay_buffer import HerReplayBuffer
from rlkit.envs.multitask.reacher_7dof import Reacher7DofXyzGoalState
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.state_distance.experimental_policies import \
    StandardTanhGaussianPolicy, OneHotTauTanhGaussianPolicy, \
    BinaryTauTanhGaussianPolicy, TauVectorTanhGaussianPolicy, \
    TauVectorSeparateFirstLayerTanhGaussianPolicy
from rlkit.torch.sac.policies import *
from rlkit.state_distance.experimental_tdm_networks import OneHotTauQF, \
    BinaryStringTauQF, TauVectorQF, TauVectorSeparateFirstLayerQF
from rlkit.state_distance.tdm_sac import TdmSac


def experiment(variant):
    env = NormalizedBoxEnv(variant['env_class']())

    obs_dim = int(np.prod(env.observation_space.low.shape))
    action_dim = int(np.prod(env.action_space.low.shape))
    vectorized = variant['algo_params']['tdm_kwargs']['vectorized']
    qf_class = variant['qf_class']
    vf_class = variant['vf_class']
    policy_class = variant['policy_class']
    qf = qf_class(
        observation_dim=obs_dim,
        action_dim=action_dim,
        goal_dim =env.goal_dim,
        output_size=env.goal_dim if vectorized else 1,
        **variant['qf_params']
    )
    vf = vf_class(
        observation_dim=obs_dim,
        goal_dim=env.goal_dim,
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
    algorithm = TdmSac(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        replay_buffer=replay_buffer,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local_docker"
    exp_prefix = "tdm-reacher_7dof-SAC"

    num_epochs = 100
    num_steps_per_epoch = 1000
    num_steps_per_eval = 1000
    max_path_length = 50
    max_tau = max_path_length-1
    # noinspection PyTypeChecker
    versions = [
        (StructuredQF, StructuredQF, StandardTanhGaussianPolicy, '_standard'),
        (OneHotTauQF, OneHotTauQF, OneHotTauTanhGaussianPolicy, '_one_hot_tau'),
        (BinaryStringTauQF, BinaryStringTauQF, BinaryTauTanhGaussianPolicy, '_binary_string_tau'),
        (TauVectorQF, TauVectorQF, TauVectorTanhGaussianPolicy, '_tau_vector'),
        (TauVectorSeparateFirstLayerQF, TauVectorSeparateFirstLayerQF, TauVectorSeparateFirstLayerTanhGaussianPolicy,
         '_separate_first_layer')
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
            sac_kwargs=dict(
                soft_target_tau=0.01,
                policy_lr=3E-4,
                qf_lr=3E-4,
                vf_lr=3E-4,
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
    )
    search_space = {
        'env_class': [
            Reacher7DofXyzGoalState,
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
            1,
            5,
            10,
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
            for i in range(n_seeds):
                seed = random.randint(0, 10000)
                run_experiment(
                    experiment,
                    seed=seed,
                    variant=variant,
                    exp_id=exp_id,
                    exp_prefix=exp_prefix+exp,
                    mode=mode,
                )
