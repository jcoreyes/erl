import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.contextual_rig_launcher_util import (
    goal_conditioned_sac_experiment, process_args
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants

from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in

if __name__ == "__main__":
    variant = dict(
        env_id='SawyerPushNIPS-v0',
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        sac_trainer_kwargs=dict(
            reward_scale=1,
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        max_path_length=100,
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=1001,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.3,
            fraction_distribution_context=0.5,
            max_size=int(1e6),
        ),
        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=10,
            pad_color=0,
        ),
        train_vae_kwargs=dict(
            vae_path=None,
            representation_size=4,
            beta=10.0 / 128,
            beta_schedule_kwargs=dict(
                x_values=(0, 500),
                y_values=(1  / 128.0, 50  / 128.0),
            ),
            num_epochs=501,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=10000,
                oracle_dataset=False,
                oracle_dataset_using_set_to_goal=False,
                non_presampled_goal_img_is_garbage=False,
                random_rollout_data=True,
                random_rollout_data_set_to_goal=True,
                conditional_vae_dataset=False,
                save_trajectories=False,
                enviorment_dataset=False,
                use_cached=False,
                vae_dataset_specific_kwargs=dict(
                ),
                show=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                is_auto_encoder=False,
                batch_size=128,
                lr=1e-3,
            ),
            save_period=5,
        ),
        renderer_kwargs=dict(
            transpose=True,
            flatten=True,
            init_camera=sawyer_init_camera_zoomed_in,
        ),
        evaluation_goal_sampling_mode="reset_of_env",
        exploration_goal_sampling_mode="vae_prior",

        launcher_config=dict(
            unpack_variant=True,
        )
    )

    search_space = {
        "seed": range(5),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(goal_conditioned_sac_experiment, variants, process_args)
