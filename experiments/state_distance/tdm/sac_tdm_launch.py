import random

import numpy as np

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.envs.multitask.point2d import MultitaskPoint2DEnv
from railrl.envs.multitask.reacher_7dof import (
    # Reacher7DofGoalStateEverything,
    Reacher7DofFullGoal)
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.state_distance.tdm_sac import TdmSac
from railrl.torch.networks import FlattenMlp


def experiment(variant):
    env = NormalizedBoxEnv(variant['env_class']())

    obs_dim = int(np.prod(env.observation_space.low.shape))
    action_dim = int(np.prod(env.action_space.low.shape))
    vectorized = variant['sac_tdm_kwargs']['tdm_kwargs']['vectorized']
    qf = FlattenMlp(
        input_size=obs_dim + action_dim + env.goal_dim + 1,
        output_size=env.goal_dim if vectorized else 1,
        **variant['qf_params']
    )
    vf = FlattenMlp(
        input_size=obs_dim + env.goal_dim + 1,
        output_size=env.goal_dim if vectorized else 1,
        **variant['vf_params']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + env.goal_dim + 1,
        action_dim=action_dim,
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
        **variant['sac_tdm_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-sac-tdm-launch"

    n_seeds = 1
    mode = "ec2"
    exp_prefix = "reacher7dof-sac-mtau-1-or-10-terminal-bonus"

    num_epochs = 100
    num_steps_per_epoch = 1000
    num_steps_per_eval = 1000
    max_path_length = 100

    # noinspection PyTypeChecker
    variant = dict(
        sac_tdm_kwargs=dict(
            base_kwargs=dict(
                num_epochs=num_epochs,
                num_steps_per_epoch=num_steps_per_epoch,
                num_steps_per_eval=num_steps_per_eval,
                max_path_length=max_path_length,
                num_updates_per_env_step=25,
                batch_size=128,
                discount=1,
                save_replay_buffer=False,
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
            max_size=int(1E6),
            num_goals_to_sample=4,
        ),
        qf_params=dict(
            hidden_sizes=[300, 300],
        ),
        vf_params=dict(
            hidden_sizes=[300, 300],
        ),
        policy_params=dict(
            hidden_sizes=[300, 300],
        ),
        version="SAC-TDM",
        algorithm="SAC-TDM",
    )
    search_space = {
        'env_class': [
            # GoalXVelHalfCheetah,
            # Reacher7DofXyzGoalState,
            Reacher7DofFullGoal,
            # MultitaskPoint2DEnv,
            # GoalXYPosAnt,
            # Walker2DTargetXPos,
            # MultitaskPusher3DEnv,
            # CylinderXYPusher2DEnv,
        ],
        'sac_tdm_kwargs.base_kwargs.reward_scale': [
            1,
            10,
            100,
            1000,
        ],
        'sac_tdm_kwargs.tdm_kwargs.vectorized': [
            # False,
            True,
        ],
        'sac_tdm_kwargs.tdm_kwargs.terminate_when_goal_reached': [
            True,
            # False,
        ],
        'sac_tdm_kwargs.tdm_kwargs.sample_rollout_goals_from': [
            # 'fixed',
            # 'environment',
            'replay_buffer',
        ],
        'relabel': [
            True,
        ],
        'sac_tdm_kwargs.tdm_kwargs.dense_rewards': [
            False,
        ],
        'sac_tdm_kwargs.tdm_kwargs.finite_horizon': [
            True,
        ],
        'sac_tdm_kwargs.tdm_kwargs.reward_type': [
            'distance',
        ],
        'sac_tdm_kwargs.tdm_kwargs.max_tau': [
            10,
            1,
            # 99,
            # 49,
            # 15,
        ],
        'sac_tdm_kwargs.tdm_kwargs.tau_sample_strategy': [
            # 'all_valid',
            'uniform',
            # 'no_resampling',
        ],
        'sac_tdm_kwargs.base_kwargs.num_updates_per_env_step': [
            1,
            # 10,
            # 25,
        ],
        'sac_tdm_kwargs.base_kwargs.discount': [
            1,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        dense = variant['sac_tdm_kwargs']['tdm_kwargs']['dense_rewards']
        finite = variant['sac_tdm_kwargs']['tdm_kwargs']['finite_horizon']
        discount = variant['sac_tdm_kwargs']['base_kwargs']['discount']
        relabel = variant['relabel']
        if not finite:
            variant['sac_tdm_kwargs']['base_kwargs']['discount'] = min(
                0.95, discount
            )
        if not dense and not finite:  # This setting makes no sense
            continue
        variant['multitask'] = (
                variant['sac_tdm_kwargs']['tdm_kwargs'][
                    'sample_rollout_goals_from'
                ] != 'fixed'
        )
        if relabel:
            variant['sac_tdm_kwargs']['tdm_kwargs']['sample_train_goals_from'] = 'her'
            variant['sac_tdm_kwargs']['tdm_kwargs'][
                'tau_sample_strategy'] = 'uniform'
        else:
            variant['sac_tdm_kwargs']['tdm_kwargs']['sample_train_goals_from'] = 'no_resampling'
            variant['sac_tdm_kwargs']['tdm_kwargs'][
                'tau_sample_strategy'] = 'no_resampling'
        for i in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                seed=seed,
                variant=variant,
                exp_id=exp_id,
            )
