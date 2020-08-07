import rlkit.misc.hyperparameter as hyp

from rlkit.data_management.her_replay_buffer import RelabelingReplayBuffer
from rlkit.envs.mujoco.sawyer_gripper_env import SawyerXYEnv
from rlkit.envs.mujoco.sawyer_reach_env import SawyerReachXYEnv
from multiworld.envs.pygame.point2d import Point2DWallEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_reset_env import SawyerPushAndReachXYZEnv

from rlkit.launchers.experiments.ashvin.multiworld import her_td3_experiment

from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants

import numpy as np

if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=501,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=100,
            num_updates_per_env_step=1,
            batch_size=128,
            discount=0.99,
            min_num_steps_before_training=128,
        ),
        # env_class=SawyerReachXYEnv,
        env_class=SawyerPushAndReachXYZEnv,
        env_kwargs=dict(
            # ball_radius=1.0,
            # render_onscreen=True,
            randomize_puck_position_on_reset=False,
            reward_type="puck_distance",
        ),
        replay_buffer_class=RelabelingReplayBuffer,
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.2,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        normalize=True,
        algorithm='HER-TD3',
        version='normal',
        observation_key='observation',
        desired_goal_key='desired_goal',
        exploration_type='ou',
        es_kwargs=dict(),
        snapshot_gap=100,
        snapshot_mode='gap_and_last',
    )

    n_seeds = 5
    search_space = {
        # 'algo_kwargs.num_updates_per_env_step': [
        #     1,
        # ],
        # 'replay_buffer_kwargs.fraction_resampled_goals_are_env_goals': [
        #     0.0,
        #     0.5,
        #     1.0,
        # ],
        # 'replay_buffer_kwargs.fraction_goals_are_rollout_goals': [
        #     0.2,
        #     # 1.0,
        # ],
        # 'env_kwargs.reward_info.type': [
        #     # 'hand_only',
        #     # 'shaped',
        #     'euclidean',
        # ],
        # 'exploration_type': [
        #     'epsilon',
        #     'ou',
        #     'gaussian',
        # ],
        'seed_id': range(n_seeds),
        'replay_buffer_kwargs.fraction_resampled_goals_are_env_goals': [
            0.0,
            0.5,
        ],
        'replay_buffer_kwargs.fraction_goals_are_rollout_goals': [
            0.2,
            1.0,
        ],
        'env_kwargs.randomize_puck_position_on_reset': [False, True],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    run_variants(her_td3_experiment, sweeper.iterate_hyperparameters(), run_id=1)
