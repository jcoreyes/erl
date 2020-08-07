from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv

from rlkit.launchers.launcher_util import run_experiment
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.experiments.murtaza.rfeatures_rl import state_td3bc_experiment

x_var = 0.2
x_low = -x_var
x_high = x_var
y_low = 0.5
y_high = 0.7
t = 0.05

if __name__ == "__main__":
    variant = dict(
        env_class=SawyerMultiobjectEnv,
        env_kwargs=dict(
            fixed_start=True,  # CHECK
            # reset_frequency=5, #CHECK
            fixed_colors=True,
            num_objects=1,
            object_meshes=None,
            num_scene_objects=[1],
            maxlen=0.1,
            action_repeat=1,
            puck_goal_low=(x_low + 0.01, y_low + 0.01),
            puck_goal_high=(x_high - 0.01, y_high - 0.01),
            hand_goal_low=(x_low + 3 * t, y_low + t),
            hand_goal_high=(x_high - 3 * t, y_high - t),
            mocap_low=(x_low + 2 * t, y_low, 0.0),
            mocap_high=(x_high - 2 * t, y_high, 0.5),
            object_low=(x_low + 0.01, y_low + 0.01, 0.02),
            object_high=(x_high - 0.01, y_high - 0.01, 0.02),
            use_textures=False,
            init_camera=sawyer_init_camera_zoomed_in,
        ),
        eval_env_kwargs=dict(
            fixed_start=True,  # CHECK
            # reset_frequency=1, #CHECK
            fixed_colors=True,
            num_objects=1,
            object_meshes=None,
            num_scene_objects=[1],
            maxlen=0.1,
            action_repeat=1,
            puck_goal_low=(x_low + 0.01, y_low + 0.01),
            puck_goal_high=(x_high - 0.01, y_high - 0.01),
            hand_goal_low=(x_low + 3 * t, y_low + t),
            hand_goal_high=(x_high - 3 * t, y_high - t),
            mocap_low=(x_low + 2 * t, y_low, 0.0),
            mocap_high=(x_high - 2 * t, y_high, 0.5),
            object_low=(x_low + 0.01, y_low + 0.01, 0.02),
            object_high=(x_high - 0.01, y_high - 0.01, 0.02),
            use_textures=False,
            init_camera=sawyer_init_camera_zoomed_in,
        ),
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=300,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=4000,
            min_num_steps_before_training=1000,
            max_path_length=100,
        ),
        td3_trainer_kwargs=dict(
            discount=0.99,
        ),
        td3_bc_trainer_kwargs=dict(
            discount=0.99,
            demo_path=None,
            demo_off_policy_path=None,
            bc_num_pretrain_steps=10000,
            q_num_pretrain_steps=10000,
            rl_weight=1.0,
            bc_weight=0,
            reward_scale=1.0,
            target_update_period=2,
            policy_update_period=2,
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        exploration_noise=.2,
        algorithm='HER-TD3',
        load_demos=False,
        pretrain_rl=False,
        pretrain_policy=False,
        es='gauss_eps',
        save_video=False,
        image_env_kwargs=dict(
            imsize=48,
            init_camera=sawyer_init_camera_zoomed_in,
        ),
        save_video_period=50,
        td3_bc=True,
    )

    search_space = {
        'exploration_noise': [.1, .2, .3, .5, .8],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_name = 'test'

    n_seeds = 6
    mode = 'ec2'
    exp_name = 'pusher_multiobj_state_td3_sweep_exp_noise'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                state_td3bc_experiment,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                num_exps_per_instance=1,
                skip_wait=False,
            )
