import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import init_sawyer_camera_v1
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_online_vae_full_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place \
        import SawyerPickAndPlaceEnv, SawyerPickAndPlaceEnvYZ
from experiments.steven.goal_generation.pickup_goal_dataset import \
        generate_vae_dataset

from multiworld.envs.mujoco.cameras import \
        sawyer_pick_and_place_camera, sawyer_pick_and_place_camera_slanted_angle
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import \
        get_image_presampled_goals
from rlkit.torch.grill.launcher import grill_her_twin_sac_online_vae_full_experiment

if __name__ == "__main__":
    num_images = 2
    variant = dict(
        imsize=48,
        double_algo=False,
        env_kwargs=dict(
            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.2),
            hide_goal_markers=True,
            random_init=True,
        ),

        env_class=SawyerPickAndPlaceEnv,

        init_camera=[sawyer_pick_and_place_camera],
        grill_variant=dict(
            save_video=True,
            save_video_period=50,
            online_vae_beta=1.0,
            presample_image_goals_only=True,
            generate_goal_dataset_fctn=get_image_presampled_goals,
            goal_generation_kwargs=dict(
                num_presampled_goals=1000,
            ),
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=2000,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=50,
                    discount=0.99,
                    num_updates_per_env_step=2,
                    # collection_mode='online-parallel',
                ),
                twin_sac_kwargs=dict(
                    train_policy_with_reparameterization=True,
                    soft_target_tau=1e-3,  # 1e-2
                    policy_update_period=1,
                    target_update_period=1,  # 1
                    use_automatic_entropy_tuning=True,
                ),
                her_kwargs=dict(),
                online_vae_kwargs=dict(),
            ),
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            replay_buffer_kwargs=dict(
                max_size=int(100000),
                fraction_goals_are_rollout_goals=0.0,
                fraction_resampled_goals_are_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='image_bernoulli_inv_prob',
                power=1,

            ),
            algorithm='GRILL-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.3,
            exploration_type='uniform_for_k',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            dump_video_kwargs=dict(
                num_images=num_images,
            )
        ),
        train_vae_variant=dict(
            generate_vae_data_fctn=generate_vae_dataset,
            dump_skew_debug_plots=False,
            representation_size=8,
            beta=0.5,
            num_epochs=0,
            generate_vae_dataset_kwargs=dict(
                N=100,
                test_p=.1,
                show=False,
                oracle_dataset=True,
                use_cached=False,
                num_channels=3*num_images,
                vae_dataset_specific_env_kwargs=dict(
                    p_obj_in_hand=.1,
                ),
            ),
            vae_kwargs=dict(
                input_channels=3*num_images,
                decoder_activation="sigmoid",
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                lr=1e-3,
                # skew_config=dict(
                    # method='inv_bernoulli_p_x',
                    # power=1,
                # ),
                # skew_dataset=True,
                # normalize_log_probs=False,
                # normalize_mean=False,
                # normalize_std=False,
                # normalize_max=False,

            ),
            #beta_schedule_kwargs=dict(
            #    x_values=[0, 100, 200, 500],
            #    y_values=[0, 0, 5, 5],
            #),
            save_period=5,
        ),
    )

    search_space = {
        'grill_variant.training_mode': ['train'],
        'grill_variant.replay_buffer_kwargs.fraction_goals_are_rollout_goals': [0.0],
        'grill_variant.online_vae_beta': [.25],
        'grill_variant.algo_kwargs.base_kwargs.num_updates_per_env_step': [2],
        'grill_variant.algo_kwargs.online_vae_kwargs.vae_training_schedule':
            [vae_schedules.every_other],
        'grill_variant.exploration_noise': [0.0],
        'grill_variant.algo_kwargs.online_vae_kwargs.oracle_data': [False],
        'grill_variant.algo_kwargs.online_vae_kwargs.parallel_vae_train': [False],
        'env_kwargs.random_init': [False],
        'env_kwargs.reset_free': [True],
        'env_kwargs.action_scale': [.02],
        'init_camera': [
            [sawyer_pick_and_place_camera_slanted_angle, sawyer_pick_and_place_camera],
            # [sawyer_pick_and_place_camera],
        ],
        'grill_variant.replay_buffer_kwargs.vae_priority_type':['image_bernoulli_inv_prob'],
        'env_kwargs.hand_high': [(0.0, 0.65, 0.2), (0.1, 0.65, 0.3)],
        'grill_variant.num_uniform_steps':[0],
        'grill_variant.algo_kwargs.base_kwargs.min_num_steps_before_training':[10000],
        'grill_variant.replay_buffer_kwargs.power': [0, 1, 2, 4],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 5
    mode = 'ec2'
    exp_prefix = 'pickup-online-vae-bernoulli-priority-more-frequent-final-4-bias=true'
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_twin_sac_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                # trial_dir_suffix='n1000-{}--zoomed-{}'.format(n1000, zoomed),
                snapshot_gap=200,
                snapshot_mode='gap_and_last',
                num_exps_per_instance=2,
            )
