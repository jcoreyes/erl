from rlkit.launchers.experiments.murtaza.multiworld import her_td3_experiment
import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEnv
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_multiple_objects import MultiSawyerEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants

import numpy as np

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=1001,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=4,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=4000,
                reward_scale=100,
                render=False,
                collection_mode='online',
                parallel_env_params=dict(
                    num_workers=1,
                ),
            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='state_desired_goal',
            ),
            td3_kwargs=dict(),
        ),
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
        algorithm='HER-TD3',
        version='normal',
        es_kwargs=dict(
            max_sigma=.8,
        ),
        exploration_type='ou',
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        init_camera=sawyer_pusher_camera_upright_v2,
        do_state_exp=True,

        save_video=True,
        imsize=84,

        snapshot_mode='gap_and_last',
        snapshot_gap=50,

        env_class=MultiSawyerEnv,
        env_kwargs=dict(
            do_render=False,
            finger_sensors=False,
            num_objects=1,
            object_meshes=None,
            randomize_initial_pos=False,
            fix_z=True,
            fix_gripper=True,
            fix_rotation=True,
            cylinder_radius = 0.03,
            maxlen = 0.03,
        ),

        num_exps_per_instance=1,
    )

    search_space = {
        # 'env_id': ['SawyerPushAndReacherXYEnv-v0', ],
        'seedid': range(3),
        'algo_kwargs.base_kwargs.num_updates_per_env_step': [4, ],
        'workspace_size': [0.05, 0.1],
        'replay_buffer_kwargs.fraction_goals_are_rollout_goals': [0.2, 1.0],
        'replay_buffer_kwargs.fraction_resampled_goals_are_env_goals': [0.0, 0.5],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )


    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer_pusher_state_final'

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        size = variant["workspace_size"]
        low = (-size, 0.4 - size, 0)
        high = (size, 0.4 + size, 0.1)
        variant["env_kwargs"]["workspace_low"] = low
        variant["env_kwargs"]["workspace_high"] = high
        variant["env_kwargs"]["hand_low"] = low
        variant["env_kwargs"]["hand_high"] = high
        variants.append(variant)

    run_variants(her_td3_experiment, variants, run_id=0)
    # for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
    #     for i in range(n_seeds):
    #         run_experiment(
    #             her_td3_experiment,
    #             exp_prefix=exp_prefix,
    #             mode=mode,
    #             snapshot_mode='gap_and_last',
    #             snapshot_gap=50,
    #             variant=variant,
    #             use_gpu=True,
    #             num_exps_per_instance=5,
    #         )
