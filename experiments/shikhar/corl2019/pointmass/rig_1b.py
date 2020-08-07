from multiworld.envs.pygame.multiobject_pygame_env import Multiobj2DEnv
import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.skew_fit.reacher.generate_uniform_dataset import generate_uniform_dataset_reacher
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_twin_sac_online_vae_full_experiment, grill_her_twin_sac_full_experiment
from rlkit.torch.grill.launcher import *
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.torch.vae.conv_vae import imsize48_default_architecture
from rlkit.launchers.arglauncher import run_variants
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv
from multiworld.envs.pygame.point2d import Point2DWallEnv

# 1a
# def experiment(variant):
#     full_experiment_variant_preprocess(variant)
#     train_vae_and_update_variant(variant)

if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        init_camera=sawyer_init_camera_zoomed_in,
        # env_id='SawyerPushNIPSEasy-v0',

		env_class=Multiobj2DEnv,
		env_kwargs=dict(
			render_onscreen=False,
			ball_radius=1,
			images_are_rgb=True,
			show_goal=False,
			fixed_colors=True,
			num_colors=7,
		),

        grill_variant=dict(
            save_video=True,
            custom_goal_sampler='replay_buffer',
            online_vae_trainer_kwargs=dict(
                beta=20,
                lr=0,
            ),
            save_video_period=100,
            qf_kwargs=dict(
                hidden_sizes=[100, 100],
            ),
            policy_kwargs=dict(
                hidden_sizes=[100, 100],
            ),
            vf_kwargs=dict(
                hidden_sizes=[100, 100],
            ),
            max_path_length=10,
            algo_kwargs=dict(
                batch_size=128,
                num_epochs=1000,
                num_eval_steps_per_epoch=100,
                num_expl_steps_per_train_loop=100,
                num_trains_per_train_loop=5,
                min_num_steps_before_training=100,
                vae_training_schedule=vae_schedules.never_train,
                oracle_data=False,
                vae_save_period=25,
                parallel_vae_train=False,
            ),
            twin_sac_trainer_kwargs=dict(
                discount=0.98,
                reward_scale=1,
                soft_target_tau=1e-3,
                target_update_period=1,  # 1
                use_automatic_entropy_tuning=True,
            ),
            replay_buffer_kwargs=dict(
                start_skew_epoch=10,
                max_size=int(100000),
                fraction_goals_rollout_goals=1.0,
                fraction_goals_env_goals=0.0,
                exploration_rewards_type='None',
				ob_keys_to_save=['state_achieved_goal', 'state_desired_goal', 'desired_goal', 'achieved_goal'],
                vae_priority_type='vae_prob',
                priority_function_kwargs=dict(
                    sampling_method='importance_sampling',
                    decoder_distribution='gaussian_identity_variance',
                    # decoder_distribution='bernoulli',
                    num_latents_to_sample=10,
                ),
                power=-1,
                relabeling_goal_sampling_mode='env',
            ),
            exploration_goal_sampling_mode='reset_of_env',
            evaluation_goal_sampling_mode='reset_of_env',
            normalize=False,
            render=False,
            exploration_noise=0.2,
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
            vae_path="shikhar/corl2019/offline/pointmass-rig2/run1/id0/vae.pkl",
        ),
        train_vae_variant=dict(
            representation_size=4,
            beta=20,
            num_epochs=10,
            dump_skew_debug_plots=False,
            # decoder_activation='gaussian',
            decoder_activation='sigmoid',
            generate_vae_dataset_kwargs=dict(
                N=10,
                test_p=.9,
                use_cached=False,
                show=False,
                oracle_dataset=False,
                oracle_dataset_using_set_to_goal=False,
                n_random_steps=2,
                non_presampled_goal_img_is_garbage=False,
                random_rollout_data=True
            ),
            vae_kwargs=dict(
                input_channels=3,
                architecture=imsize48_default_architecture,
                decoder_distribution='gaussian_identity_variance',
            ),
            # TODO: why the redundancy?
            algo_kwargs=dict(
                start_skew_epoch=5000,
                is_auto_encoder=False,
                batch_size=64,
                lr=1e-3,
                skew_config=dict(
                    method='vae_prob',
                    power=0,
                ),
                skew_dataset=False,
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
    )

    search_space = {
        'seedid': range(3),
        'grill_variant.vae_path': [
			'shikhar/corl2019/pointmass/cvae/run0/id0/vae.pkl',
			'shikhar/corl2019/pointmass/cvae/run0/id1/vae.pkl',
			'shikhar/corl2019/pointmass/cvae/run0/id2/vae.pkl',
			'shikhar/corl2019/pointmass/cvae/run0/id3/vae.pkl',
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(grill_her_twin_sac_online_vae_full_experiment, variants, run_id=0)
