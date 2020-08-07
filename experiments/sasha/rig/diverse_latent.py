import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.skew_fit.reacher.generate_uniform_dataset import generate_uniform_dataset_reacher
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
# from rlkit.torch.grill.launcher import grill_her_twin_sac_online_vae_full_experiment
from rlkit.torch.grill.launcher import *
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.torch.vae.conv_vae import imsize48_default_architecture
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.vae.conv_vae import ConvDynamicsVAE
from rlkit.torch.grill.launcher import grill_her_td3_full_experiment
from multiworld.envs.pygame.point2d import Point2DWallEnv

def experiment(variant):
    grill_her_twin_sac_full_experiment(variant)
    #full_experiment_variant_preprocess(variant)
    #train_vae_and_update_variant(variant)


if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        init_camera=sawyer_init_camera_zoomed_in,

        grill_variant=dict(
            save_video=True,
            save_video_period=100,
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
                    num_updates_per_env_step=4,
                    collection_mode='online',
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
                fraction_goals_rollout_goals=0.1,
                fraction_goals_env_goals=0.5,
            ),
            algorithm='RIG-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.2,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance'
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
            vae_path=None,
        ),

        env_class=Point2DWallEnv,
        env_kwargs=dict(
            render_onscreen=False,
            ball_radius=1,
            images_are_rgb=True,
            show_goal=False,
        ),

        train_vae_variant=dict(
            use_linear_dynamics=True,
            representation_size=4,
            vae_class=ConvDynamicsVAE,
            beta=10,
            num_epochs=10,
            dump_skew_debug_plots=False,
            # decoder_activation='gaussian',
            decoder_activation='sigmoid',
            generate_vae_dataset_kwargs=dict(
                N=1000,
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
                dynamics_type='global',
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
                skew_dataset=False,
                linearity_weight=10,
                distance_weight=10,
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

    region="us-west-1",

    search_space = {
        'seedid': range(1),
        'train_vae_variant.beta': [1],
        'train_vae_variant.algo_kwargs.linearity_weight': [1],
        'train_vae_variant.algo_kwargs.distance_weight': [0.1]

        # 'seedid': range(3),
        # 'train_vae_variant.beta': [1, 10, 20, 50],
        # 'train_vae_variant.algo_kwargs.linearity_weight': [0, 1, 10, 50, 100],
        # 'train_vae_variant.algo_kwargs.distance_weight': [0, 0.1, 1, 10, 50, 100]
        # 'train_vae_variant.vae_kwargs.dynamics_type': ['global', 'local', 'nonlinear']
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, run_id=1)