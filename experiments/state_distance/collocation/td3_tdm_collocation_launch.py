import random
import torch

import numpy as np
from torch.nn import functional as F

import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.her_replay_buffer import HerReplayBuffer
from rlkit.envs.multitask.point2d import MultitaskPoint2DEnv
from rlkit.envs.multitask.point2d_wall import MultitaskPoint2dWall
from rlkit.envs.multitask.reacher_7dof import (
    # Reacher7DofGoalStateEverything,
    Reacher7DofFullGoal, Reacher7DofXyzGoalState)
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.state_distance.tdm_networks import TdmNormalizer, TdmQf, \
    TdmVf, StochasticTdmPolicy, TdmPolicy
from rlkit.state_distance.experimental_tdm_networks import DebugQf
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.mpc.collocation.collocation_mpc_controller import (
    TdmLBfgsBCMC,
    TdmToImplicitModel, LBfgsBCMC, TdmLBfgsBStateOnlyCMC)
from rlkit.torch.mpc.controller import MPCController, DebugQfToMPCController
from rlkit.state_distance.tdm_td3 import TdmTd3
from rlkit.torch.networks import FlattenMlp


def experiment(variant):
    vectorized = variant['td3_tdm_kwargs']['tdm_kwargs']['vectorized']
    env = NormalizedBoxEnv(variant['env_class'](**variant['env_kwargs']))
    max_tau = variant['td3_tdm_kwargs']['tdm_kwargs']['max_tau']
    qf1 = TdmQf(
        env,
        vectorized=vectorized,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
        env,
        vectorized=vectorized,
        **variant['qf_kwargs']
    )
    tdm_normalizer = TdmNormalizer(
        env,
        vectorized,
        max_tau=max_tau,
        **variant['tdm_normalizer_kwargs']
    )
    implicit_model = TdmToImplicitModel(
        env,
        qf1,
        tau=0,
    )
    vf = TdmVf(
        env=env,
        vectorized=vectorized,
        tdm_normalizer=tdm_normalizer,
        **variant['vf_kwargs']
    )
    policy = TdmPolicy(
        env=env,
        tdm_normalizer=tdm_normalizer,
        **variant['policy_kwargs']
    )
    replay_buffer = HerReplayBuffer(
        env=env,
        **variant['her_replay_buffer_kwargs']
    )
    goal_slice = env.ob_to_goal_slice
    lbfgs_mpc_controller = TdmLBfgsBCMC(
        implicit_model,
        env,
        goal_slice=goal_slice,
        multitask_goal_slice=goal_slice,
        tdm_policy=policy,
        **variant['mpc_controller_kwargs']
    )
    state_only_mpc_controller = TdmLBfgsBStateOnlyCMC(
        vf,
        policy,
        env,
        goal_slice=goal_slice,
        multitask_goal_slice=goal_slice,
        **variant['state_only_mpc_controller_kwargs']
    )
    es = GaussianStrategy(
        action_space=env.action_space,
        **variant['es_kwargs']
    )
    if variant['explore_with'] =='TdmLBfgsBCMC':
        raw_exploration_policy = lbfgs_mpc_controller
    elif variant['explore_with'] =='TdmLBfgsBStateOnlyCMC':
        raw_exploration_policy = state_only_mpc_controller
    else:
        raw_exploration_policy = policy
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=raw_exploration_policy,
    )
    if variant['eval_with'] == 'TdmLBfgsBCMC':
        eval_policy = lbfgs_mpc_controller
    elif variant['eval_with'] == 'TdmLBfgsBStateOnlyCMC':
        eval_policy = state_only_mpc_controller
    else:
        eval_policy = policy
    # variant['td3_tdm_kwargs']['base_kwargs']['eval_policy'] = eval_policy
    algorithm = TdmTd3(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        exploration_policy=exploration_policy,
        eval_policy=eval_policy,
        replay_buffer=replay_buffer,
        **variant['td3_tdm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-td3-tdm-launch"

    # n_seeds = 3
    # mode = "ec2"
    exp_prefix = "real-td3-tdm-lbfgs-dynamic-lm"

    num_epochs = 50
    num_steps_per_epoch = 100
    num_steps_per_eval = 100
    max_path_length = 50

    # noinspection PyTypeChecker
    variant = dict(
        td3_tdm_kwargs=dict(
            base_kwargs=dict(
                num_epochs=num_epochs,
                num_steps_per_epoch=num_steps_per_epoch,
                num_steps_per_eval=num_steps_per_eval,
                max_path_length=max_path_length,
                min_num_steps_before_training=128,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=1,
                save_replay_buffer=False,
                render=True,
            ),
            tdm_kwargs=dict(
                sample_rollout_goals_from='environment',
                sample_train_goals_from='her',
                vectorized=True,
                norm_order=2,
                cycle_taus_for_rollout=True,
                max_tau=0,
                square_distance=True,
                reward_type='distance',
                terminate_when_goal_reached=True,
            ),
            td3_kwargs=dict(),
        ),
        her_replay_buffer_kwargs=dict(
            max_size=int(1E6),
            num_goals_to_sample=4,
        ),
        qf_kwargs=dict(
            hidden_sizes=[32, 32],
            hidden_activation=torch.tanh,
            learn_offset=False,
            structure='squared_difference',
        ),
        vf_kwargs=dict(
            hidden_sizes=[32, 32],
            structure='squared_difference',
        ),
        policy_kwargs=dict(
            hidden_sizes=[32, 32],
        ),
        tdm_normalizer_kwargs=dict(
            normalize_tau=False,
            log_tau=False,
        ),
        mpc_controller_kwargs=dict(
            lagrange_multipler=10,
            planning_horizon=3,
            replan_every_time_step=True,
            only_use_terminal_env_loss=True,
            dynamic_lm=True,
            solver_kwargs={
                'factr': 1e12,
            },
        ),
        state_only_mpc_controller_kwargs=dict(
            lagrange_multipler=10,
            planning_horizon=3,
            replan_every_time_step=True,
            only_use_terminal_env_loss=True,
            solver_kwargs={
                'factr': 1e10,
            },
        ),
        es_kwargs=dict(
            max_sigma=0.5,
        ),
        env_kwargs=dict(),
        version="TD3-TDM",
        algorithm="TD3-TDM",
    )
    search_space = {
        'env_class': [
            # GoalXVelHalfCheetah,
            # Reacher7DofXyzGoalState,
            # Reacher7DofFullGoal,
            # MultitaskPoint2DEnv,
            MultitaskPoint2dWall,
            # GoalXYPosAnt,
            # Walker2DTargetXPos,
            # MultitaskPusher3DEnv,
            # CylinderXYPusher2DEnv,
        ],
        'td3_tdm_kwargs.base_kwargs.reward_scale': [
            1,
        ],
        'td3_tdm_kwargs.tdm_kwargs.max_tau': [
            0,
        ],
        'eval_with': [
            'TdmLBfgsBCMC',
            # 'TdmLBfgsBStateOnlyCMC',
            # 'none',
        ],
        'explore_with': [
            'TdmLBfgsBCMC',
            # 'TdmLBfgsBStateOnlyCMC',
            # 'none',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                variant=variant,
                exp_id=exp_id,
                # snapshot_mode='gap',
                # snapshot_gap=5,
            )
