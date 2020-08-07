import rlkit.misc.hyperparameter as hyp
from rlkit.images.camera import (
    sawyer_init_camera_zoomed_in_fixed,
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEnv
)
from rlkit.torch.networks.experimental import HuberLoss
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_tdm_td3_online_vae_full_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules

if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        # env_class=SawyerReachXYEnv,
        env_class=SawyerPushAndReachXYEnv,
        # env_class=SawyerPickAndPlaceEnv,
        env_kwargs=dict(
            hide_goal_markers=True,
            action_scale=.02,
            puck_low=[-0.25, .4],
            puck_high=[0.25, .8],
            mocap_low=[-0.2, 0.45, 0.],
            mocap_high=[0.2, 0.75, 0.5],
            goal_low=[-0.2, 0.45, 0.02, -0.25, 0.4],
            goal_high=[0.2, 0.75, 0.02, 0.25, 0.8],
        ),
        init_camera=sawyer_init_camera_zoomed_in_fixed,
        grill_variant=dict(
            save_video=True,
            save_video_period=25,
            online_vae_beta=1.0,
            algo_kwargs=dict(
                tdm_td3_kwargs=dict(
                    base_kwargs=dict(
                        num_epochs=3000,
                        num_steps_per_epoch=1000,
                        num_steps_per_eval=1000,
                        max_path_length=100,
                        batch_size=256,
                        discount=1,
                        min_num_steps_before_training=4000,
                    ),
                    tdm_kwargs=dict(
                    ),
                    td3_kwargs=dict(
                        tau=1,
                    ),
                ),
                online_vae_algo_kwargs=dict(
                    vae_training_schedule=vae_schedules.every_three,
                ),
            ),


            replay_kwargs=dict(
                max_size=int(40000),
                fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0.5,
                exploration_rewards_scale=0.0,
                exploration_rewards_type='None',

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

            algorithm='GRILL-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.3,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
        ),
        train_vae_variant=dict(
            representation_size=8,
            beta=5.0,
            num_epochs=0,
            generate_vae_dataset_kwargs=dict(
                N=1000,
                test_p=.9,
                oracle_dataset=True,
                use_cached=False,
                num_channels=3,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                lr=1e-3,
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
        # 'grill_variant.replay_kwargs.fraction_resampled_goals_are_env_goals': [.5, 1],
        'grill_variant.replay_kwargs.fraction_goals_are_rollout_goals': [0.2],
        'grill_variant.replay_kwargs.power': [0],
        'grill_variant.algo_kwargs.online_vae_kwargs.vae_training_schedule':
        [vae_schedules.every_six],
        'grill_variant.exploration_noise': [.8],
        # 'grill_variant.exploration_type': ['ou', 'gaussian', 'epsilon'],
        'grill_variant.algo_kwargs.online_vae_kwargs.oracle_data': [False],
        'grill_variant.algo_kwargs.tdm_td3_kwargs.td3_kwargs.tau': [1],
        'grill_variant.algo_kwargs.tdm_td3_kwargs.base_kwargs.num_updates_per_env_step': [ 4],
        'grill_variant.algo_kwargs.tdm_td3_kwargs.base_kwargs.reward_scale': [1, 10, 100],
        'grill_variant.algo_kwargs.tdm_td3_kwargs.tdm_kwargs.max_tau': [ 20],
        'grill_variant.algo_kwargs.tdm_td3_kwargs.tdm_kwargs.vectorized': [False],
        'grill_variant.qf_kwargs.structure': ['none'],

    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 2
    mode = 'local'
    exp_prefix = 'pusher-online-vae-tdm-larger-goal-sweep'

    # n_seeds = 3
    # mode = 'ec2'
    # exp_prefix = 'multiworld-goalenv-full-grill-her-td3'
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_tdm_td3_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                # trial_dir_suffix='n1000-{}--zoomed-{}'.format(n1000, zoomed),
                snapshot_gap=200,
                snapshot_mode='gap_and_last',
                num_exps_per_instance=2,
            )
