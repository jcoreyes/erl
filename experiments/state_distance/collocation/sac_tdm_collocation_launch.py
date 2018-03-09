import random

import numpy as np
from torch.nn import functional as F

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.envs.multitask.point2d import MultitaskPoint2DEnv
from railrl.envs.multitask.point2d_wall import MultitaskPoint2dWall
from railrl.envs.multitask.reacher_7dof import (
    # Reacher7DofGoalStateEverything,
    Reacher7DofFullGoal, Reacher7DofXyzGoalState)
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.state_distance.tdm_networks import TdmNormalizer, TdmQf, \
    TdmVf, StochasticTdmPolicy
from railrl.state_distance.experimental_tdm_networks import DebugQf
from railrl.torch.data_management.normalizer import TorchFixedNormalizer
from railrl.torch.mpc.collocation.collocation_mpc_controller import (
    TdmLBfgsBCMC,
    TdmToImplicitModel, LBfgsBCMC)
from railrl.torch.mpc.controller import MPCController, DebugQfToMPCController
from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.state_distance.tdm_sac import TdmSac
from railrl.torch.networks import FlattenMlp


def experiment(variant):
    vectorized = variant['sac_tdm_kwargs']['tdm_kwargs']['vectorized']
    env = NormalizedBoxEnv(variant['env_class'](**variant['env_kwargs']))
    max_tau = variant['sac_tdm_kwargs']['tdm_kwargs']['max_tau']
    qf = DebugQf(
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
        qf,
        tau=0,
    )
    goal_slice = env.ob_to_goal_slice
    lbfgs_mpc_controller = TdmLBfgsBCMC(
    # lbfgs_mpc_controller = LBfgsBCMC(
        implicit_model,
        env,
        goal_slice=goal_slice,
        multitask_goal_slice=goal_slice,
        **variant['mpc_controller_kwargs']
    )
    if variant['explore_with'] =='TdmLBfgsBCMC':
        variant['sac_tdm_kwargs']['base_kwargs']['exploration_policy'] = (
            lbfgs_mpc_controller
        )
    if variant['eval_with'] == 'TdmLBfgsBCMC':
        variant['sac_tdm_kwargs']['base_kwargs']['eval_policy'] = (
            lbfgs_mpc_controller
        )
    vf = TdmVf(
        env=env,
        vectorized=vectorized,
        tdm_normalizer=tdm_normalizer,
        **variant['vf_kwargs']
    )
    policy = StochasticTdmPolicy(
        env=env,
        tdm_normalizer=tdm_normalizer,
        **variant['policy_kwargs']
    )
    replay_buffer = HerReplayBuffer(
        env=env,
        **variant['her_replay_buffer_kwargs']
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

    # n_seeds = 3
    # mode = "ec2"
    # exp_prefix = "reacher7dof-xyz-refactor"

    num_epochs = 50
    num_steps_per_epoch = 100
    num_steps_per_eval = 100
    max_path_length = 50

    # noinspection PyTypeChecker
    variant = dict(
        sac_tdm_kwargs=dict(
            base_kwargs=dict(
                num_epochs=num_epochs,
                num_steps_per_epoch=num_steps_per_epoch,
                num_steps_per_eval=num_steps_per_eval,
                max_path_length=max_path_length,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=1,
                save_replay_buffer=False,
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
            ),
            sac_kwargs=dict(
                soft_target_tau=0.01,
                policy_lr=3E-4,
                qf_lr=3E-4,
                vf_lr=3E-4,
            ),
            give_terminal_reward=False,
        ),
        her_replay_buffer_kwargs=dict(
            max_size=int(1E6),
            num_goals_to_sample=4,
        ),
        qf_kwargs=dict(
            hidden_sizes=[300, 300],
            hidden_activation=ptu.softplus,
            predict_delta=True,
        ),
        vf_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        tdm_normalizer_kwargs=dict(
            normalize_tau=False,
            log_tau=False,
        ),
        mpc_controller_kwargs=dict(
            lagrange_multipler=100,
            planning_horizon=3,
            replan_every_time_step=True,
            solver_kwargs={
                'factr': 1e12,
            },
        ),
        env_kwargs=dict(),
        version="SAC-TDM",
        algorithm="SAC-TDM",
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
        'sac_tdm_kwargs.base_kwargs.reward_scale': [
            1,
        ],
        'sac_tdm_kwargs.give_terminal_reward': [
            False,
        ],
        'sac_tdm_kwargs.tdm_kwargs.terminate_when_goal_reached': [
            # True,
            False,
        ],
        'sac_tdm_kwargs.tdm_kwargs.max_tau': [
            0,
        ],
        'eval_with': [
            # 'DebugQfToMPCController',
            'TdmLBfgsBCMC',
            # 'TanhGaussianPolicy',
        ],
        'explore_with': [
            # 'DebugQfToMPCController',
            'TdmLBfgsBCMC',
            # 'TanhGaussianPolicy',
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
