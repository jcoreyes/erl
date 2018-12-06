import railrl.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.fit_skew.pick_and_place.generate_uniform_dataset import \
    generate_uniform_dataset_pick_and_place
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
from railrl.envs.goal_generation.pickup_goal_dataset import get_image_presampled_goals_from_vae_env
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.grill.launcher import grill_her_twin_sac_online_vae_full_experiment
import railrl.torch.vae.vae_schedules as vae_schedules
from railrl.torch.vae.conv_vae import imsize48_default_architecture
from railrl.envs.goal_generation.pickup_goal_dataset import \
        generate_vae_dataset

if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        env_id="SawyerPickupEnv-v0",
        init_camera=sawyer_pick_and_place_camera,
        grill_variant=dict(
            save_video=True,
            online_vae_beta=2.5,
            save_video_period=250,
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
                    num_epochs=1010,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=10000,
                    batch_size=128,
                    max_path_length=100,
                    discount=0.99,
                    num_updates_per_env_step=2,
                    collection_mode='online-parallel',
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
                   vae_training_schedule=vae_schedules.every_other,
                    oracle_data=False,
                    vae_save_period=100,
                    parallel_vae_train=False,
                ),
            ),
            replay_buffer_kwargs=dict(
                max_size=int(100000),
                fraction_goals_are_rollout_goals=0,
                fraction_resampled_goals_are_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='image_bernoulli_inv_prob',
                priority_function_kwargs=dict(
                    sampling_method='correct',
                    num_latents_to_sample=10,
                    decode_prob='none',
                ),
                power=2,
            ),
            normalize=False,
            render=False,
            exploration_noise=0,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            presample_goals=True,
            generate_goal_dataset_fctn=get_image_presampled_goals_from_vae_env,
            goal_generation_kwargs=dict(
                num_presampled_goals=1000,
            ),
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
            algorithm='ONLINE-VAE-SAC-BERNOULLI-HER-TD3',
            generate_uniform_dataset_kwargs=dict(
                env_id="SawyerPickupEnv-v0",
                init_camera=sawyer_pick_and_place_camera,
                num_imgs=1000,
                use_cached_dataset=False,
            ),
            generate_uniform_dataset_fn=generate_uniform_dataset_pick_and_place,
        ),
        train_vae_variant=dict(
            dump_skew_debug_plots=False,
            generate_vae_data_fctn=generate_vae_dataset,
            representation_size=16,
            beta=1.0,
            num_epochs=0,
            generate_vae_dataset_kwargs=dict(
                N=50,
                test_p=.9,
                oracle_dataset=True,
                show=False,
                use_cached=True,
                num_channels=3,
            ),
            vae_kwargs=dict(
                input_channels=3,
                architecture=imsize48_default_architecture,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                lr=1e-3,
            ),
            decoder_activation='sigmoid',
            save_period=5,
        ),
    )

    search_space = {
        'grill_variant.online_vae_beta':[1, 2.5],
        'grill_variant.replay_buffer_kwargs.power':[1, 2, 4],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'test'

    # n_seeds = 5
    # mode = 'gcp'
    # exp_prefix = 'pickup_online_vae_bernoulli'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_twin_sac_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=2,
                gcp_kwargs=dict(
                    zone='us-west2-b',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-p4',
                        num_gpu=1,
                    )
                )
          )
