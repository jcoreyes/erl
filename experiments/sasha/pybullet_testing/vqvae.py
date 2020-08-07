import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.contextual.rig.rig_launcher import (
    rig_experiment, process_args
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.vae.vq_vae import VQ_VAE
from rlkit.torch.vae.vq_vae_trainer import VQ_VAETrainer
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from roboverse.envs.sawyer_rig_grasp_v1 import SawyerRigGraspV1Env

if __name__ == "__main__":
    x_var = 0.2
    x_low = -x_var
    x_high = x_var
    y_low = 0.5
    y_high = 0.7
    t = 0.05

    variant = dict(
        imsize=48,
        env_class=SawyerRigGraspV1Env,
        env_kwargs=dict(
            obs_img_dim=48,
        ),
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
            save_video_period=50,
            pad_color=0,
            subpad_length=1,
            pad_length=1,
            num_columns_per_rollout=2,

        ),
        # pretrained_vae_path="ashvin/contexts/rig/old1/run0/id0/vae.pkl",
        train_vae_kwargs=dict(
            vae_class=VQ_VAE,
            vae_trainer_class=VQ_VAETrainer,
            vae_path=None,
            embedding_dim=3,
            beta=10.0 / 128,
            beta_schedule_kwargs=dict(
                x_values=(0, 500),
                y_values=(1, 50),
            ),
            num_epochs=501,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=1000,
                n_random_steps=50,
                oracle_dataset_using_set_to_goal=False,
                non_presampled_goal_img_is_garbage=True,
                random_rollout_data=True,
                random_rollout_data_set_to_goal=False,
                conditional_vae_dataset=True,
                save_trajectories=True,
                dataset_path="/home/ashvin/data/sasha/demos/alt_encoder_images.npy",
                enviorment_dataset=False,
                use_cached=False,
                tag="contextual1",
                vae_dataset_specific_kwargs=dict(
                ),
                show=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
                imsize=48,
            ),
            algo_kwargs=dict(
                key_to_reconstruct='x_t',
                do_scatterplot=False,
                use_linear_dynamics=False,
                is_auto_encoder=False,
                batch_size=256,
                lr=1e-3,
                weight_decay=0.0,
            ),
            save_period=25,
        ),
        renderer_kwargs=dict(
            #input_img_format='HWC',
            #output_img_format='CWH',
            #flatten_img=True,
        ),
        evaluation_goal_sampling_mode="reset_of_env",
        exploration_goal_sampling_mode="vae_prior",

        launcher_config=dict(
            unpack_variant=True,
        )
    )

    search_space = {
        "seed": range(1),
        'train_vae_kwargs.embedding_dim': [3, 2],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(rig_experiment, variants, process_args, run_id=10)
