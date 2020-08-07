import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.skew_fit.reacher.generate_uniform_dataset import generate_uniform_dataset_reacher
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in, sawyer_pusher_camera_upright_v2
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import *
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.torch.vae.conv_vae import imsize48_default_architecture, imsize48_default_architecture_with_more_hidden_layers, imsize84_default_architecture
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.grill.launcher import grill_her_twin_sac_online_vae_full_experiment, grill_her_twin_sac_full_experiment
from multiworld.envs.pygame.multiobject_pygame_env import Multiobj2DEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv
from rlkit.torch.vae.conditional_conv_vae import DeltaCVAE
from rlkit.torch.vae.vae_trainer import DeltaCVAETrainer

def experiment(variant):
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)


x_var = 0.2
x_low = -x_var
x_high = x_var
y_low = 0.5
y_high = 0.7
t = 0.05

if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        init_camera=sawyer_init_camera_zoomed_in,

        env_class=SawyerMultiobjectEnv,
        env_kwargs=dict(
            num_objects=10,
            object_meshes=None,
            fixed_start=False,
            num_scene_objects=[1],
            maxlen=0.1,
            action_repeat=5,
            puck_goal_low=(x_low, y_low),
            puck_goal_high=(x_high, y_high),
            hand_goal_low=(x_low, y_low),
            hand_goal_high=(x_high, y_high),
            mocap_low=(x_low, y_low, 0.0),
            mocap_high=(x_high, y_high, 0.5),
            object_low=(x_low + t + t, y_low + t, 0.02),
            object_high=(x_high - t - t, y_high - t, 0.02),
            preload_obj_dict=[
                dict(color2=(1, 0, 0)),
                dict(color2=(0, 1, 0)),
                dict(color2=(0, 0, 1)),
                dict(color2=(1, 1, 1)),
                dict(color2=(1, 0, 1)),
                dict(color2=(0, 1, 1)),
                dict(color2=(1, 1, 0)),
                dict(color2=(.4, 0, .4)),
                dict(color2=(.4, .2, 0)),
                dict(color2=(0, .4, .4)),
            ],
        ),
        # env_class=Multiobj2DEnv,
        # env_kwargs=dict(
        #     render_onscreen=False,
        #     ball_radius=1.5,
        #     images_are_rgb=True,
        #     show_goal=False,
        #     change_background=False,
        #     fixed_colors=False,
        # ),

        grill_variant=dict(
            save_video=True,
            custom_goal_sampler='replay_buffer',
            online_vae_trainer_kwargs=dict(
                beta=20,
                lr=0,
            ),
            save_video_period=100,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            vf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            max_path_length=50,
            algo_kwargs=dict(
                num_epochs=2000,
                batch_size=128,
                num_eval_steps_per_epoch=1000,
                num_expl_steps_per_train_loop=1000,
                num_trains_per_train_loop=500,
                min_num_steps_before_training=1000,
                vae_training_schedule=vae_schedules.never_train,
                oracle_data=False,
                vae_save_period=25,
                parallel_vae_train=False,
            ),
            twin_sac_trainer_kwargs=dict(
                discount=0.99,
                reward_scale=1,
                soft_target_tau=1e-3,
                target_update_period=1,  # 1
                use_automatic_entropy_tuning=True,
            ),
            replay_buffer_kwargs=dict(
                start_skew_epoch=10,
                max_size=int(100000),
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
                power=-1,
                relabeling_goal_sampling_mode='vae_prior',
            ),
            exploration_goal_sampling_mode='vae_prior',
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
        ),
        train_vae_variant=dict(
            lr=1e-4,
            batch_size=32,
            representation_size=4,
            beta=10,
            num_epochs=2501,
            dump_skew_debug_plots=False,
            # decoder_activation='gaussian',
            decoder_activation='sigmoid',
            use_linear_dynamics=False,
            generate_vae_dataset_kwargs=dict(
                dataset_path="/home/shikharbahl/new_data.npy",
                N=17000,
                n_random_steps=10,
                test_p=.9,
                use_cached=False,
                show=False,
                oracle_dataset=False,
                oracle_dataset_using_set_to_goal=False,
                non_presampled_goal_img_is_garbage=False,
                random_rollout_data=True,
                conditional_vae_dataset=True,
                save_trajectories=False,
                enviorment_dataset=False,
            ),
            vae_trainer_class=DeltaCVAETrainer,
            vae_class=DeltaCVAE,
            vae_kwargs=dict(
#                dynamics_type='local',
                input_channels=3,
                architecture=imsize48_default_architecture_with_more_hidden_layers,
                decoder_distribution='gaussian_identity_variance',
            ),
            # TODO: why the redundancy?
            algo_kwargs=dict(
                start_skew_epoch=5000,
                is_auto_encoder=False,
                batch_size=128,
                lr=1e-4,
                skew_config=dict(
                    method='vae_prob',
                    power=0,
                ),
                linearity_weight=0,
                distance_weight=0,
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
        'seedid': range(1),
        'train_vae_variant.representation_size': [(8, 20)], #(3 * objects, 3 * colors)
        'train_vae_variant.beta': [50], #THIS IS A LINEAR INTERPOLATION CURRENTLY
        # 'train_vae_variant.generate_vae_dataset_kwargs.n_random_steps': [2],#, 50, 100],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, run_id=100)
