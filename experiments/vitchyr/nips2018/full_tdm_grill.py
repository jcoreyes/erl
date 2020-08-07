import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import init_sawyer_camera_v1, \
    init_sawyer_camera_v2, init_sawyer_camera_v4
from multiworld.envs.pygame.point2d import Point2DEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEnv
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_full_experiment, \
    grill_tdm_td3_full_experiment
from rlkit.torch.networks.experimental import HuberLoss
from rlkit.torch.vae.sawyer2d_push_variable_data import generate_vae_dataset

if __name__ == "__main__":
    variant = dict(
        # env_class=SawyerReachXYEnv,
        env_class=SawyerPushAndReachXYEnv,
        # env_class=SawyerPickAndPlaceEnv,
        # env_class=Point2DEnv,
        env_kwargs=dict(
            hide_goal_markers=True,
            # puck_low=(-0.15, 0.5),
            # puck_high=(0.15, 0.7),
            puck_low=(-0.2, 0.5),
            puck_high=(0.2, 0.7),
            hand_low=(-0.2, 0.5, 0.),
            hand_high=(0.2, 0.7, 0.5),
            mocap_low=(-0.1, 0.5, 0.),
            mocap_high=(0.1, 0.7, 0.5),
            goal_low=(-0.05, 0.55, 0.02, -0.2, 0.5),
            goal_high=(0.05, 0.65, 0.02, 0.2, 0.7),
        ),
        init_camera=init_sawyer_camera_v4,
        grill_variant=dict(
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=250,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    max_path_length=100,
                    num_updates_per_env_step=4,
                    batch_size=128,
                    discount=1,
                    min_num_steps_before_training=1000,
                ),
                tdm_kwargs=dict(
                    max_tau=15,
                ),
                td3_kwargs=dict(
                    tau=1,
                ),
            ),
            replay_kwargs=dict(
                max_size=int(3e5),
                fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0.5,
            ),
            vae_wrapped_env_kwargs=dict(
                num_goals_presampled=1000,
            ),
            algorithm='GRILL-TDM-TD3',
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
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
                structure='none',
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            qf_criterion_class=HuberLoss,
            # vae_path='06-25-pusher-state-puck-reward-cached-goals-hard-2/06-25-pusher-state-puck-reward-cached-goals-hard-2-id0-s48265/vae.pkl',
            vae_path="05-23-vae-sawyer-variable-fixed-2/05-23-vae-sawyer-variable-fixed-2_2018_05_23_16_19_33_0000--s-293-nImg-1000--cam-sawyer_init_camera_zoomed_in_fixed/params.pkl",
            # vae_path="06-28-train-vae-beta-5-push-and-reach-cam4-2/06-28-train-vae-beta-5-push-and-reach-cam4-2_2018_06_28_11_47_21_0000--s-11654/params.pkl",
            # vae_path="06-28-train-vae-beta-5-push-and-reach-cam4-p15-range/06-28-train-vae-beta-5-push-and-reach-cam4-p15-range_2018_06_28_11_48_04_0000--s-80805/params.pkl",
        ),
        train_vae_variant=dict(
            representation_size=16,
            beta=5.0,
            num_epochs=1000,
            generate_vae_dataset_kwargs=dict(
                # dataset_path='manual-upload/SawyerPushAndReachXYEnv_1000_init_sawyer_camera_v4_oracleTrue.npy',
                N=1000,
                oracle_dataset=True,
                num_channels=3,
                show=True,
                use_cached=False,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                lr=1e-3,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            beta_schedule_kwargs=dict(
                x_values=[0, 100, 200, 500],
                y_values=[0, 0, 5, 5],
            ),
            save_period=5,
        ),
    )

    search_space = {
        'grill_variant.algo_kwargs.tdm_kwargs.max_tau': [15],
        'grill_variant.algo_kwargs.base_kwargs.reward_scale': [
            100,
        ],
        'grill_variant.algo_kwargs.base_kwargs.max_path_length': [
            100,
        ],
        'hand-goal-space': ['easy', 'hard'],
        'mocap-x-range': ['0.1'],
        'grill_variant.do_state_exp': [False],
        'grill_variant.vae_path': [
            # None,
            # "06-28-train-vae-beta-5-push-and-reach-cam4-2/06-28-train-vae-beta-5-push-and-reach-cam4-2_2018_06_28_11_47_21_0000--s-11654/params.pkl",
            "05-23-vae-sawyer-variable-fixed-2/05-23-vae-sawyer-variable-fixed-2_2018_05_23_16_19_33_0000--s-293-nImg-1000--cam-sawyer_init_camera_zoomed_in_fixed/params.pkl",
        ],
        # 'grill_variant.replay_kwargs.fraction_goals_are_rollout_goals': [
        #     0.2,
        # ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    mode = 'local'
    exp_prefix = 'dev'

    mode = 'ec2'
    exp_prefix = 'gw-vitchyr-check'
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if variant['hand-goal-space'] == 'easy':
            variant['env_kwargs']['goal_low'] = (-0.05, 0.55, 0.02, -0.2, 0.5)
            variant['env_kwargs']['goal_high'] = (0.05, 0.65, 0.02, 0.2, 0.7)
        else:
            variant['env_kwargs']['goal_low'] = (-0.2, 0.5, 0.02, -0.2, 0.5)
            variant['env_kwargs']['goal_high'] = (0.2, 0.7, 0.02, 0.2, 0.7)
        if variant['mocap-x-range'] == '0.1':
            variant['env_kwargs']['mocap_low'] = (-0.1, 0.5, 0.)
            variant['env_kwargs']['mocap_high'] = (0.1, 0.7, 0.5)
        else:
            variant['env_kwargs']['mocap_low'] = (-0.2, 0.5, 0.)
            variant['env_kwargs']['mocap_high'] = (0.2, 0.7, 0.5)
        run_experiment(
            grill_tdm_td3_full_experiment,
            exp_prefix=exp_prefix,
            mode=mode,
            variant=variant,
            use_gpu=True,
            # snapshot_gap=50,
            snapshot_mode='last',
            exp_id=exp_id,
            num_exps_per_instance=3,
        )
