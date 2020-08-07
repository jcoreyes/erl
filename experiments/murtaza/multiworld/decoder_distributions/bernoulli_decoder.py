import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v0
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_full_experiment
from rlkit.torch.vae.conv_vae import ConvVAE
from rlkit.torch.vae.dataset.generate_goal_dataset import generate_goal_dataset_using_policy

architecture = dict(
        conv_args=dict(
            kernel_sizes=[5, 3, 3],
            n_channels=[16, 32, 64],
            strides=[3, 2, 2],
        ),
        conv_kwargs=dict(
            hidden_sizes=[500, 300, 150],
            batch_norm_conv=False,
            batch_norm_fc=False,
        ),
        deconv_args=dict(
            hidden_sizes=[150, 300, 500],

            deconv_input_width=3,
            deconv_input_height=3,
            deconv_input_channels=64,

            deconv_output_kernel_size=6,
            deconv_output_strides=3,
            deconv_output_channels=3,

            kernel_sizes=[3, 3],
            n_channels=[32, 16],
            strides=[2, 2],
        ),
        deconv_kwargs=dict(
            batch_norm_deconv=False,
            batch_norm_fc=False,
        )
    )

if __name__ == "__main__":
    variant = dict(
        imsize=48,
        init_camera=sawyer_door_env_camera_v0,
        env_id='SawyerDoorHookEnv-v0',
        grill_variant=dict(
            save_video=True,
            save_video_period=50,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=505,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=100,
                    discount=0.99,
                    num_updates_per_env_step=1,
                    collection_mode='online-parallel',
                    parallel_env_params=dict(
                        num_workers=1,
                    ),
                    reward_scale=1,
                ),
                her_kwargs=dict(),
                td3_kwargs=dict(
                    tau=1e-2,
                ),
            ),
            replay_buffer_kwargs=dict(
                max_size=int(1e6),
                fraction_goals_are_rollout_goals=0,
                fraction_resampled_goals_are_env_goals=0.5,
            ),
            algorithm='OFFLINE-VAE-RECON-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.3,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            generate_goal_dataset_fctn=generate_goal_dataset_using_policy,
            goal_generation_kwargs=dict(
                num_goals=1000,
                use_cached_dataset=False,
                path_length=100,
                policy_file='11-09-sawyer-door-state-her-td3/11-09-sawyer_door_state_her_td3_2018_11_09_19_17_28_id000--s92604/params.pkl',
                show=False,
            ),
            presample_goals=True,
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            )
        ),
        train_vae_variant=dict(
            vae_path=None,
            representation_size=16,
            beta=.5,
            num_epochs=1000,
            dump_skew_debug_plots=False,
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=5000,
                use_cached=True,
                oracle_dataset_from_policy=True,
                random_and_oracle_policy_data=True,
                non_presampled_goal_img_is_garbage=True,
                random_and_oracle_policy_data_split=0,
                vae_dataset_specific_kwargs=dict(),
                policy_file='11-09-sawyer-door-state-her-td3/11-09-sawyer_door_state_her_td3_2018_11_09_19_17_28_id000--s92604/params.pkl',
                show=False,
            ),
            vae_class=ConvVAE,
            vae_kwargs=dict(
                input_channels=3,
            ),
            decoder_activation='sigmoid',
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                is_auto_encoder=False,
                batch_size=64,
                lr=1e-3,
            ),
            save_period=10,
        ),
    )

    search_space = {
        'train_vae_variant.beta': [.5, 1, 2.5],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 3
    mode = 'gcp'
    exp_prefix = 'sawyer_door_offline_vae_bernoulli'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_td3_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=1,
                gcp_kwargs=dict(
                    zone='us-west1-a',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-p100',
                        num_gpu=1,
                    )
                )
          )
