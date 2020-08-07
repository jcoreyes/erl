import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.skew_fit.reacher.generate_uniform_dataset import generate_uniform_dataset_reacher
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_twin_sac_online_vae_full_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.torch.vae.conv_vae import imsize48_default_architecture

if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerPushNIPSEasy-v0',
        grill_variant=dict(
            sample_goals_from_buffer=True,
            save_video=False,
            online_vae_beta=20,
            save_video_period=50,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            vf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=1000,
                    num_steps_per_epoch=500,
                    num_steps_per_eval=500,
                    min_num_steps_before_training=10000,
                    batch_size=256,
                    max_path_length=50,
                    discount=0.99,
                    num_updates_per_env_step=2,
                    # collection_mode='online-parallel',
                    parallel_env_params=dict(
                        num_workers=1,
                    ),
                    reward_scale=1,
                ),

                her_kwargs=dict(
                ),
                twin_sac_kwargs=dict(
                    train_policy_with_reparameterization=True,
                    soft_target_tau=1e-3,  # 1e-2
                    policy_update_period=1,
                    target_update_period=1,  # 1
                    use_automatic_entropy_tuning=True,
                ),
                online_vae_kwargs=dict(
                    vae_training_schedule=vae_schedules.custom_schedule_2,
                    oracle_data=False,
                    vae_save_period=50,
                    parallel_vae_train=False,
                ),
            ),
            replay_buffer_kwargs=dict(
                start_skew_epoch=10,
                max_size=int(10000),
                fraction_goals_rollout_goals=0.2,
                fraction_goals_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='vae_prob',
                priority_function_kwargs=dict(
                    sampling_method='importance_sampling',
                    decoder_distribution='gaussian_identity_variance',
                    # decoder_distribution='bernoulli',
                    num_latents_to_sample=10,
                ),
                power=.1,

            ),
            normalize=False,
            render=False,
            exploration_noise=0.0,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
            algorithm='ONLINE-VAE-SAC-BERNOULLI',
            # generate_uniform_dataset_kwargs=dict(
                # init_camera=sawyer_init_camera_zoomed_in,
                # env_id='SawyerPushNIPS-v0',
                # num_imgs=1000,
                # use_cached_dataset=False,
                # show=False,
                # save_file_prefix='pusher',
            # ),
            # generate_uniform_dataset_fn=generate_uniform_dataset_reacher,
        ),
        train_vae_variant=dict(
            representation_size=4,
            beta=20,
            num_epochs=0,
            dump_skew_debug_plots=False,
            decoder_activation='gaussian',
            # decoder_activation='sigmoid',
            generate_vae_dataset_kwargs=dict(
                N=40,
                test_p=.9,
                use_cached=True,
                show=False,
                oracle_dataset=True,
                oracle_dataset_using_set_to_goal=True,
                n_random_steps=100,
                non_presampled_goal_img_is_garbage=True,
            ),
            vae_kwargs=dict(
                input_channels=3,
                architecture=imsize48_default_architecture,
                decoder_distribution='gaussian_identity_variance',
            ),
            algo_kwargs=dict(
                start_skew_epoch=5000,
                is_auto_encoder=False,
                batch_size=64,
                lr=1e-3,
                skew_config=dict(
                    method='vae_prob',
                    power=0,
                ),
                skew_dataset=True,
                priority_function_kwargs=dict(
                    decoder_distribution='gaussian_identity_variance',
                    sampling_method='importance_sampling',
                    # sampling_method='true_prior_sampling',
                    num_latents_to_sample=10,
                ),
                use_parallel_dataloading=False,
            ),

            save_period=25,
        ),
        version='no force',
    )

    search_space = {
        'grill_variant.vae_wrapped_env_kwargs.goal_sampler_for_exploration': [True],
        'grill_variant.vae_wrapped_env_kwargs.goal_sampler_for_relabeling': [True],
        'grill_variant.replay_buffer_kwargs.priority_function_kwargs.num_latents_to_sample':[
            10
        ],
        'grill_variant.replay_buffer_kwargs.power': [-1.],
        'grill_variant.replay_buffer_kwargs.max_size': [
            10000,
            20000,
            40000,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 2
    mode = 'sss'
    exp_prefix = 'pusher-sf-steven-reference-script-rb-size-sweep'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_twin_sac_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=3,
                time_in_mins=int(2.8*24*60),
                snapshot_gap=100,
                snapshot_mode='gap_and_last',
                gcp_kwargs=dict(
                    terminate=True,
                    zone='us-east1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                )
          )
