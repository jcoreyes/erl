from rlkit.launchers.experiments.murtaza.multiworld import her_td3_experiment
import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEnv
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj import SawyerMultiobjectEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants

import numpy as np

if __name__ == "__main__":
    # noinspection PyTypeChecker

    x_low = -0.2
    x_high = 0.2
    y_low = 0.5
    y_high = 0.7
    t = 0.03

    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=501,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=4,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=4000,
                reward_scale=1.0,
                render=False,
                collection_mode='online',
                tau=1e-2,
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
            fraction_goals_rollout_goals=0.1,
            fraction_goals_env_goals=0.5,
            ob_keys_to_save=[],
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
            max_sigma=.2,
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

        env_class=SawyerMultiobjectEnv,
        env_kwargs=dict(
            num_objects=1,
            preload_obj_dict=[
                dict(color2=(0.1, 0.1, 0.9)),
            ],
        ),

        num_exps_per_instance=1,
        region="us-east-1",
    )

    search_space = {
        'seedid': range(4),
        'algo_kwargs.base_kwargs.num_updates_per_env_step': [16, ],
        'replay_buffer_kwargs.fraction_goals_rollout_goals': [0.1, ],
        'replay_buffer_kwargs.fraction_goals_env_goals': [0.5, ],
        'env_kwargs.action_repeat': [1, 5, 25],
        'algo_kwargs.base_kwargs.max_path_length': [10, 20, 100],
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
        variants.append(variant)

    run_variants(her_td3_experiment, variants, run_id=0)
