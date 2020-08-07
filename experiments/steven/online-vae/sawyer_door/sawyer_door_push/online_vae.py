import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorPushOpenActionLimitedEnv

from rlkit.images.camera import sawyer_door_env_camera_closer
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.vae.dataset.sawyer_door_push_open_data import generate_vae_dataset
from rlkit.torch.grill.launcher import grill_her_td3_online_vae_full_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules

if __name__ == "__main__":
#    n_seeds = 1
#    mode = 'local'
#    exp_prefix = 'test'

    n_seeds = 3
    mode = 'ec2'
    exp_prefix = 'sawyer-door-push-open-online-vae-oracle-sweep-2'

    grill_variant = dict(
        online_vae_beta=2.5,
        save_video=True,
        save_video_period=25,
        algo_kwargs=dict(
            num_epochs=4000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=2000,
            batch_size=128,
            max_path_length=200,
            discount=0.99,
            min_num_steps_before_training=5000,
            vae_training_schedule=vae_schedules.every_three,
            num_updates_per_env_step=2,
            # collection_mode='online-parallel',
            # parallel_env_params=dict(
                # num_workers=2,
            # )

        ),
        replay_kwargs=dict(
            max_size=40000,
            fraction_goals_are_rollout_goals=.2,
            fraction_resampled_goals_are_env_goals=.5,
            alpha=3,
            exploration_rewards_scale=0.0,
            exploration_rewards_type='reconstruction_error',
        ),
        algorithm='GRiLL-HER-TD3',
        normalize=False,
        render=False,
        use_env_goals=True,
        version='normal',
        exploration_type='ou',
        exploration_noise=.3,
        training_mode='train_env_goals',
        testing_mode='test',
        reward_params=dict(
            min_variance=0,
            type='latent_distance',
        ),
        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',

    )
    train_vae_variant = dict(
        generate_vae_data_fctn=generate_vae_dataset,
        beta=5,
        num_epochs=0,
        algo_kwargs=dict(
            batch_size=256,
        ),
        vae_kwargs=dict(
            min_variance=None,
            input_channels=3,
        ),
        generate_vae_dataset_kwargs=dict(
            # N=5000,
            # use_cached=False,
            # oracle_dataset=True,
            # show=False,
            # # test_p=.9,
            # num_channels=3,
            N=5000,
            # test_p=.8,
            use_cached=False,
            action_plus_random_sampling=True,
        ),
        representation_size=16,
        save_period=10,
    )
    variant = dict(
        double_algo=False,
        grill_variant=grill_variant,
        train_vae_variant=train_vae_variant,
        env_class=SawyerDoorPushOpenActionLimitedEnv,
        env_kwargs=dict(
        ),
        init_camera=sawyer_door_env_camera_closer,
    )

    search_space = {
        'grill_variant.algo_kwargs.oracle_data': [True, False],
        'grill_variant.replay_kwargs.power': [0, 1, 2],
        'train_vae_variant.representation_size': [6],
        'grill_variant.training_mode': ['train'],
        # 'grill_variant.replay_kwargs.fraction_resampled_goals_are_env_goals': [0.0, .5, 1],
        # 'grill_variant.replay_kwargs.fraction_goals_are_rollout_goals': [0.2],
        # 'grill_variant.replay_kwargs.power': [50, 10, 5, 3, 1],
        # 'grill_variant.algo_kwargs.hard_restart_period': [20000],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                grill_her_td3_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=2,
            )
