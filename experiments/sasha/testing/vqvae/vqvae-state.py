import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.skew_fit.reacher.generate_uniform_dataset import generate_uniform_dataset_reacher
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in, sawyer_pusher_camera_upright_v2
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.torch.vae.conv_vae import imsize48_default_architecture, imsize48_default_architecture_with_more_hidden_layers
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.grill.cvae_experiments import (
    grill_her_td3_offpolicy_online_vae_full_experiment,
)
from rlkit.misc.ml_util import PiecewiseLinearSchedule, ConstantSchedule
from multiworld.envs.pygame.multiobject_pygame_env import Multiobj2DEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv
from rlkit.torch.vae.vq_vae import VQ_VAE
from rlkit.torch.vae.vq_vae_trainer import VQ_VAETrainer
from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer

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
            fixed_start=True,
            fixed_colors=False,
            num_objects=1,
            object_meshes=None,
            preload_obj_dict=
            [{'color1': [1, 1, 1],
            'color2': [1, 1, 1]}],
            num_scene_objects=[1],
            maxlen=0.1,
            action_repeat=1,
            puck_goal_low=(x_low + 0.01, y_low + 0.01),
            puck_goal_high=(x_high - 0.01, y_high - 0.01),
            hand_goal_low=(x_low + 0.01, y_low + 0.01),
            hand_goal_high=(x_high - 0.01, y_high - 0.01),
            mocap_low=(x_low, y_low, 0.0),
            mocap_high=(x_high, y_high, 0.5),
            object_low=(x_low + 0.01, y_low + 0.01, 0.02),
            object_high=(x_high - 0.01, y_high - 0.01, 0.02),
            use_textures=False,
            init_camera=sawyer_init_camera_zoomed_in,
            cylinder_radius=0.05,
        ),

        grill_variant=dict(
            save_video=True,
            custom_goal_sampler='replay_buffer',
            online_vae_trainer_kwargs=dict(
                beta=20,
                lr=0,
            ),
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
            max_path_length=100,
            algo_kwargs=dict(
                batch_size=128,
                num_epochs=500,
                num_eval_steps_per_epoch=1000,
                num_expl_steps_per_train_loop=1000,
                num_trains_per_train_loop=1000,
                min_num_steps_before_training=4000,
                vae_training_schedule=vae_schedules.never_train,
                oracle_data=False,
                vae_save_period=25,
                parallel_vae_train=False,
                dataset_path=None,
                rl_offpolicy_num_training_steps=0,
            ),
            td3_trainer_kwargs=dict(
                discount=0.99,
                reward_scale=1.0,
                tau=1e-2,
            ),
            replay_buffer_class=OnlineVaeRelabelingBuffer,
            replay_buffer_kwargs=dict(
                ob_keys_to_save=['state_achieved_goal', "state_desired_goal"],
                start_skew_epoch=10,
                max_size=int(100000),
                fraction_goals_rollout_goals=0.2,
                fraction_goals_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='vae_prob',
                priority_function_kwargs=dict(
                    sampling_method='importance_sampling',
                    decoder_distribution='gaussian_identity_variance',
                    num_latents_to_sample=10,
                ),
                power=-1,
                relabeling_goal_sampling_mode='presampled',
            ),
            exploration_goal_sampling_mode='reset_of_env',
            evaluation_goal_sampling_mode='reset_of_env',
            normalize=False,
            render=False,
            exploration_noise=0.8,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                #epsilon=9,
                type="state_distance"
            ),
            observation_key="latent_achieved_goal", #'latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
                num_latents_to_sample=10000,
            ),
            algorithm='ONLINE-VAE-SAC-BERNOULLI',
            vae_path="/home/ashvin/data/rail-khazatsky/sasha/testing/vqvae/vqvae-state/sasha/testing/vqvae/vqvae-state/run51/id0/vae.pkl",
            #"/home/ashvin/data/sasha/testing/vqvae/vqvae-vqvae/run100/id0/vae.pkl"
                    ),
        train_vae_variant=dict(
            beta=10,
            num_epochs=501,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            use_linear_dynamics=False,
            generate_vae_dataset_kwargs=dict(
                N=100000,
                n_random_steps=50,
                test_p=.9,
                dataset_path="/home/ashvin/Desktop/sim_puck_data.npy",
                use_cached=False,
                show=False,
                oracle_dataset=False,
                oracle_dataset_using_set_to_goal=False,
                non_presampled_goal_img_is_garbage=False,
                random_rollout_data=True,
                random_rollout_data_set_to_goal=True,
                conditional_vae_dataset=True,
                save_trajectories=False,
                enviorment_dataset=False,
                tag="ccrig_tuning_orig_network",
            ),
            vae_trainer_class=VQ_VAETrainer,
            vae_class=VQ_VAE,
            vae_kwargs=dict(
                input_channels=3,
            ),

            algo_kwargs=dict(
                start_skew_epoch=5000,
                is_auto_encoder=False,
                batch_size=256,
                lr=1e-3,
                skew_config=dict(
                    method='vae_prob',
                    power=0,
                ),
                weight_decay=0.0,
                skew_dataset=False,
                priority_function_kwargs=dict(
                    decoder_distribution='gaussian_identity_variance',
                    sampling_method='importance_sampling',
                    num_latents_to_sample=10,
                ),
                use_parallel_dataloading=False,
            ),

            save_period=25,
        ),
        launcher_config=dict(
            region='us-east-2'
        ),

        logger_variant=dict(
            tensorboard=True,
        ),

        slurm_variant=dict(
            timeout_min=48 * 60,
            cpus_per_task=10,
            gpus_per_node=1,
        ),
    )

    search_space = {
        'seedid': range(1),
        'train_vae_variant.representation_size': [2,],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(grill_her_td3_offpolicy_online_vae_full_experiment, variants, run_id=1002)
