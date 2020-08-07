from rlkit.launchers.experiments.ashvin.multiworld import her_td3_experiment
import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in, sawyer_pusher_camera_upright_v2

import numpy as np

x_var = 0.2
x_low = -x_var
x_high = x_var
y_low = 0.5
y_high = 0.7
t = 0.05

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=1000,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=4000,
            min_num_steps_before_training=1000,
            max_path_length=100,
        ),
        trainer_kwargs=dict(),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
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
            max_sigma=0.5,
        ),
        exploration_type='ou',
        observation_key='state_observation',
        do_state_exp=True,

        save_video=True,
        save_video_period=100,
        imsize=84,
        init_camera=sawyer_init_camera_zoomed_in,

        snapshot_mode='gap_and_last',
        snapshot_gap=50,

        env_class=SawyerMultiobjectEnv,
        env_kwargs=dict(
            fixed_start=True,
            fixed_colors=False,
            reward_type="dense",
            #reward_mask="objects",
            num_objects=1,
            object_meshes=None,
            num_scene_objects=[1],
            maxlen=0.1,
            action_repeat=1,
            puck_goal_low=(x_low + 0.01, y_low + 0.01),
            puck_goal_high=(x_high - 0.01, y_high - 0.01),
            hand_goal_low=(x_low + 3*t, y_low + t),
            hand_goal_high=(x_high - 3*t, y_high -t),
            mocap_low=(x_low + 2*t, y_low , 0.0),
            mocap_high=(x_high - 2*t, y_high, 0.5),
            object_low=(x_low + 0.01, y_low + 0.01, 0.02),
            object_high=(x_high - 0.01, y_high - 0.01, 0.02),
            use_textures=False,
            init_camera=sawyer_init_camera_zoomed_in,
        ),

        wrap_mujoco_gym_to_multi_env=False,
        num_exps_per_instance=1,
        region='us-west-1',
    )

    search_space = {
        'seedid': range(3),
        'replay_buffer_kwargs.fraction_goals_rollout_goals': [0.2,],
        'replay_buffer_kwargs.fraction_goals_env_goals': [0.5, ],
        'env_kwargs.reward_mask':['objects', 'hand'],
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

    run_variants(her_td3_experiment, variants, run_id=1)
