from railrl.envs.mujoco.sawyer_push_and_reach_env import \
    SawyerPushAndReachXYEnv, SawyerPushAndReachXYEasyEnv
from railrl.envs.mujoco.sawyer_push_env import SawyerPushXYEnv
from railrl.envs.multitask.point2d import MultitaskImagePoint2DEnv
from railrl.envs.multitask.pusher2d import FullPusher2DEnv
from railrl.launchers.arglauncher import run_variants
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.grill.launcher import grill_her_td3_experiment, grill_tdm_td3_experiment, grill_tdm_td3_full_experiment
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place \
        import SawyerPickAndPlaceEnv, SawyerPickAndPlaceEnvYZ
from railrl.torch.modules import HuberLoss

if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'pick-and-place-tdm-full-grill-small-obj-zoomed-in-oracle-compare'
    # vae_path = '06-26-multiworld-pick-vae/06-26-multiworld-pick-vae_2018_06_26_23_20_15_0000--s-92887/vae.pkl'
    vae_path = "06-27-multiworld-pick-vae/06-27-multiworld-pick-vae_2018_06_27_17_14_41_0000--s-28626/vae_T.pkl"
    # confirm .2
    vae_path = "06-29-multiworld-pick-vae-smaller-goal-space-reproduce/06-29-multiworld-pick-vae-smaller-goal-space-reproduce_2018_06_29_16_44_12_0000--s-89480/vae.pkl"


    variant = dict(
        env_class=SawyerPickAndPlaceEnvYZ,
        init_camera=sawyer_pick_and_place_camera,
        env_kwargs=dict(
            hide_arm=True,
            hide_goal_markers=True,
        ),
        train_vae_variant = dict(
            beta=5.0,
            representation_size=8,
            generate_vae_dataset_kwargs=dict(
                N=10000,
                test_p=0.9,
                use_cached=False,
                imsize=84,
                num_channels=3,
                show=False,
                oracle_dataset=True,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
            ),
            beta_schedule_kwargs=dict(
                x_values=[0, 100, 200, 500, 1000],
                y_values=[0, 0, 0, 2.5, 5],
            ),
            save_period=1,
            num_epochs=1000,
        ),
        grill_variant=dict(
            render=False,
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=2000,
                    num_steps_per_epoch=500,
                    num_steps_per_eval=500,
                    max_path_length=31,
                    batch_size=256,
                    discount=1,
                    min_num_steps_before_training=1000,
                ),
                tdm_kwargs=dict(
                ),
                td3_kwargs=dict(
                    tau=1,
                ),
            ),

            replay_kwargs=dict(
                max_size=int(1e6),
                fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0.5,
            ),
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
                structure='norm_difference',
            ),
            qf_criterion_class=HuberLoss,
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            es_kwargs=dict(
            ),
            algorithm='HER-TD3',
            vae_path=vae_path,
            normalize=False,
            use_env_goals=True,
            wrap_mujoco_env=True,
            do_state_based_exp=False,
            exploration_noise=0.3,
        )
    )
    search_space = {
        'grill_variant.exploration_type': [
            'ou',
        ],
        'env_kwargs.oracle_reset_prob': [0.0, 0.5],
        'grill_variant.algo_kwargs.td3_kwargs.tau': [1],
        'grill_variant.algo_kwargs.base_kwargs.num_updates_per_env_step': [1],
        'grill_variant.algo_kwargs.base_kwargs.reward_scale': [10],
        'env_kwargs.action_scale': [.02],
        'grill_variant.replay_kwargs.fraction_resampled_goals_are_env_goals': [0.5],
        'grill_variant.replay_kwargs.fraction_goals_are_rollout_goals': [0.2],
        #'reward_params.epsilon': [.5],
        'grill_variant.exploration_noise': [0.3, .5],
        'grill_variant.algo_kwargs.tdm_kwargs.max_tau': [30],
        'grill_variant.algo_kwargs.tdm_kwargs.vectorized': [False],
        'grill_variant.qf_kwargs.structure': ['none'],
        'grill_variant.training_mode': ['train'],
        'grill_variant.testing_mode': ['test', ],

        'grill_variant.reward_params.type': [
            # 'mahalanobis_distance',
            # 'log_prob',
            'latent_distance',
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
                # grill_tdm_td3_experiment,
                grill_tdm_td3_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                # trial_dir_suffix='n1000-{}--zoomed-{}'.format(n1000, zoomed),
                snapshot_gap=50,
                snapshot_mode='gap_and_last',
                num_exps_per_instance=3,
            )
