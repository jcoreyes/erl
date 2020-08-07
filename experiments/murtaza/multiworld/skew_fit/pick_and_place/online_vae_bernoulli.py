import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.skew_fit.pick_and_place.generate_uniform_dataset import \
    generate_uniform_dataset_pick_and_place
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
from rlkit.envs.goal_generation.pickup_goal_dataset import get_image_presampled_goals_from_vae_env
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_twin_sac_online_vae_full_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.envs.goal_generation.pickup_goal_dataset import \
        generate_vae_dataset

if __name__ == "__main__":
    num_images = 1
    variant = dict(
        imsize=48,
        double_algo=False,
        env_id="SawyerPickupEnv-v0",
        init_camera=sawyer_pick_and_place_camera,
        grill_variant=dict(
            online_vae_beta=.25,
            save_video=True,
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
                    num_epochs=505,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=10000,
                    batch_size=128,
                    max_path_length=50,
                    discount=0.99,
                    num_updates_per_env_step=2,
                    collection_mode='online-parallel',
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
                    vae_save_period=100,
                    parallel_vae_train=False,
                ),
            ),
            replay_buffer_kwargs=dict(
                max_size=int(100000),
                fraction_goals_rollout_goals=0,
                fraction_goals_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='None',
                priority_function_kwargs=dict(
                    sampling_method='correct',
                    num_latents_to_sample=10,
                ),
                power=.5,
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
                sample_from_true_prior=False,
            ),
            algorithm='ONLINE-VAE-SAC-HER',
            generate_uniform_dataset_kwargs=dict(
                env_id="SawyerPickupEnv-v0",
                init_camera=sawyer_pick_and_place_camera,
                num_imgs=1000,
                use_cached_dataset=False,
            ),
            generate_uniform_dataset_fn=generate_uniform_dataset_pick_and_place,
        ),
        train_vae_variant=dict(
            generate_vae_data_fctn=generate_vae_dataset,
            dump_skew_debug_plots=False,
            representation_size=16,
            beta=0.25,
            num_epochs=0,
            generate_vae_dataset_kwargs=dict(
                N=100,
                oracle_dataset=True,
                use_cached=True,
                num_channels=3 * num_images,
            ),
            vae_kwargs=dict(
                input_channels=3 * num_images,
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
        'grill_variant.replay_buffer_kwargs.vae_priority_type': ['image_bernoulli_inv_prob'],
        'grill_variant.replay_buffer_kwargs.power': [0.25, .5, 2],
        'grill_variant.replay_buffer_kwargs.priority_function_kwargs.num_latents_to_sample': [10, 30],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 6
    mode = 'gcp'
    exp_prefix = 'pickup-skew-fit-sweep'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_twin_sac_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                snapshot_gap=200,
                snapshot_mode='gap_and_last',
                num_exps_per_instance=1,
                gcp_kwargs=dict(
                    zone='us-west2-b',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-p4',
                        num_gpu=1,
                    )
                )
          )
