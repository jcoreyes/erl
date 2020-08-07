from rlkit.envs.mujoco.sawyer_push_and_reach_env import \
    SawyerPushAndReachXYEnv, SawyerPushAndReachXYEasyEnv
from rlkit.envs.mujoco.sawyer_push_env import SawyerPushXYEnv
from rlkit.envs.multitask.point2d import MultitaskImagePoint2DEnv
from rlkit.envs.multitask.pusher2d import FullPusher2DEnv
from rlkit.images.camera import (
    # sawyer_init_camera,
    # sawyer_init_camera_zoomed_in,
    sawyer_init_camera_zoomed_in_fixed,
    sawyer_init_camera_zoomed_out_fixed,
)

from rlkit.launchers.arglauncher import run_variants
from multiworld.envs.mujoco.cameras import \
        sawyer_pick_and_place_camera, sawyer_pick_and_place_camera_slanted_angle
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_full_experiment
from experiments.steven.goal_generation.pickup_goal_dataset import \
        generate_vae_dataset

from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place \
        import SawyerPickAndPlaceEnv, SawyerPickAndPlaceEnvYZ
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import \
        get_image_presampled_goals


if __name__ == "__main__":
    n_seeds = 2
    mode = 'ec2'
    exp_prefix = 'grill-pick-and-place-forced-drop-reset-free'
    variant = dict(
        imsize=48,
        env_kwargs=dict(
            hand_low=(-0.1, .55, .05),
            hand_high=(0.0, .65, .2),
            hide_goal_markers=True,
            random_init=True,
        ),

        env_class=SawyerPickAndPlaceEnv,
        # init_camera=sawyer_pick_and_place_camera,
        train_vae_variant = dict(
            # spatial_vae=True,
            beta=5.0,
            representation_size=8,
            generate_vae_data_fctn=generate_vae_dataset,
            generate_vae_dataset_kwargs=dict(
                N=8000,
                use_cached=True,
                num_channels=6,
                show=False,
                oracle_dataset=True,
            ),
            vae_kwargs=dict(
                input_channels=6,
            ),
            algo_kwargs=dict(
            ),
            beta_schedule_kwargs=dict(
                x_values=[0, 100, 200, 500, 1000],
                y_values=[0, 0, 0, 2.5, 2.5],
            ),
            save_period=5,
            num_epochs=1000,
        ),
        grill_variant = dict(
            dump_video_kwargs=dict(
                num_images=2,
            ),

            presample_image_goals_only=True,
            generate_goal_dataset_fn=get_image_presampled_goals,
            goal_generation_kwargs=dict(
                num_presampled_goals=1000,
            ),
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=3000,
                    num_steps_per_epoch=500,
                    num_steps_per_eval=500,
                    batch_size=128,
                    max_path_length=100,
                    discount=0.99,
                    min_num_steps_before_training=1000,
                    # collection_mode='online-parallel',
                    # parallel_env_params=dict(
                        # num_workers=20,
                    # )
                ),
                her_kwargs=dict(),
                td3_kwargs=dict(
                    tau=1e-2,
                ),
            ),
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            replay_kwargs=dict(
                max_size=int(1e6),
                fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0.5,
            ),
            algorithm='HER-TD3',
            # vae_path="08-06-grill-pick-and-place-yz/08-06-grill-pick-and-place-yz_2018_08_06_12_54_01_0000--s-55708/vae.pkl",
            # vae_path="08-21-grill-pick-and-place-more-tests-2/08-21-grill-pick-and-place-more-tests-2_2018_08_21_19_00_27_0000--s-15955/vae.pkl",
            normalize=False,
            render=False,
            use_env_goals=True,
            wrap_mujoco_env=True,
            do_state_based_exp=True,
            exploration_noise=0.3,
        )
    )

    search_space = {
        'grill_variant.exploration_type': [
            'ou'
        ],
        'grill_variant.replay_kwargs.fraction_resampled_goals_are_env_goals': [.5],
        'train_vae_variant.representation_size': [6],
        'grill_variant.algo_kwargs.base_kwargs.num_updates_per_env_step': [2],
        'env_kwargs.oracle_reset_prob': [0.0],
        'env_kwargs.random_init': [False],
        'env_kwargs.reset_free': [True],
        'env_kwargs.action_scale': [.02],
        'train_vae_variant.vae_dataset_specific_env_kwargs':
        [
            dict(obj_in_hand=.65),
            # dict(obj_in_hand=.8),
        ],
        'env_kwargs.hand_low': [(-0.1, .55, .05)],
        'grill_variant.exploration_noise': [.4, .8],
        'grill_variant.algo_kwargs.base_kwargs.reward_scale': [1],
        'grill_variant.reward_params.type': [
            'latent_distance',
        ],
        'grill_variant.training_mode': ['train'],
        'grill_variant.testing_mode': ['test', ],
        'init_camera': [
            [sawyer_pick_and_place_camera_slanted_angle, sawyer_pick_and_place_camera],
            # sawyer_pick_and_place_camera_slanted_angle
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        # if variant['init_camera'] == sawyer_init_camera_zoomed_in:
        #     variant['vae_paths']['16'] = zoomed_in_path
        # elif variant['init_camera'] == sawyer_init_camera:
        #     variant['vae_paths']['16'] = zoomed_out_path
        # zoomed = 'zoomed_out' not in variant['vae_paths']['16']
        # n1000 = 'nImg-1000' in variant['vae_paths']['16']
        # if zoomed:
            # variant['init_camera'] = sawyer_init_camera_zoomed_out_fixed
        for _ in range(n_seeds):
            run_experiment(
                grill_her_td3_full_experiment,
                #grill_her_td3_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                # trial_dir_suffix='n1000-{}--zoomed-{}'.format(n1000, zoomed),
                snapshot_gap=100,
                snapshot_mode='gap_and_last',
                num_exps_per_instance=3,
            )
