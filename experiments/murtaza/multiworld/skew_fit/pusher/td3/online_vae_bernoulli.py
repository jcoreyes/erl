import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2, \
    sawyer_pusher_camera_upright_v1
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_online_vae_full_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.torch.vae.conv_vae import imsize48_default_architecture

if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        grill_variant=dict(
            save_video=True,
            online_vae_beta=2.5,
            save_video_period=50,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=3010,
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
                td3_kwargs=dict(
                    tau=1e-2,
                ),
                online_vae_kwargs=dict(
                    vae_training_schedule=vae_schedules.every_other,
                    oracle_data=False,
                    vae_save_period=50,
                    parallel_vae_train=False,
                ),
            ),
            replay_buffer_kwargs=dict(
                max_size=int(100000),
                fraction_goals_are_rollout_goals=0.5,
                fraction_resampled_goals_are_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='image_bernoulli_inv_prob',
                power=1,
            ),
            normalize=False,
            render=False,
            exploration_noise=0.8,
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
            algorithm='ONLINE-VAE-TD3-BERNOULLI',
        ),
        train_vae_variant=dict(
            representation_size=16,
            beta=1.0,
            num_epochs=0,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            generate_vae_dataset_kwargs=dict(
                N=100,
                test_p=.9,
                use_cached=False,
                show=False,
                oracle_dataset=False,
                n_random_steps=1,
                non_presampled_goal_img_is_garbage=True,
            ),
            vae_kwargs=dict(
                input_channels=3,
                architecture=imsize48_default_architecture,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                lr=1e-3,
            ),
            save_period=1,
        ),
    )

    search_space = {
        'grill_variant.replay_buffer_kwargs.vae_priority_type':['image_bernoulli_inv_prob'],
        'env_id': ['SawyerPushAndReachSmallArenaEnv-v0', 'SawyerPushAndReachArenaEnv-v0'],
        'init_camera':[sawyer_pusher_camera_upright_v1, sawyer_pusher_camera_upright_v2]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'


    n_seeds = 3
    mode = 'gcp'
    exp_prefix = 'pusher_online_vae_bernoulli_td3'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if variant['init_camera'] == sawyer_pusher_camera_upright_v1 and variant['env_id'] == 'SawyerPushAndReachArenaEnv-v0':
            continue
        elif variant['init_camera'] == sawyer_pusher_camera_upright_v2 and variant['env_id'] == 'SawyerPushAndReachSmallArenaEnv-v0':
            continue
        for _ in range(n_seeds):
            run_experiment(
                grill_her_td3_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=2,
                gcp_kwargs=dict(
                    zone='us-east1-b',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-p100',
                        num_gpu=1,
                    )
                )
          )
