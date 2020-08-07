import rlkit.misc.hyperparameter as hyp
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorPushOpenActionLimitedEnv
from rlkit.images.camera import (
    sawyer_door_env_camera_closer)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_full_experiment
from rlkit.torch.vae.dataset.sawyer_door_push_open_data import generate_vae_dataset
# from rlkit.torch.vae.sawyer_door_push_and_reach_data import generate_vae_dataset

if __name__ == "__main__":
    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'sawyer_door_push_open_sweep_offline-rdim6'
    grill_variant = dict(
        save_video=True,
        save_video_period=25,
        algo_kwargs=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=2000,
            max_path_length=200,
            batch_size=128,
            discount=0.99,
            reward_scale=1,
        ),
        replay_buffer_class=ObsDictRelabelingBuffer,
        replay_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=1,
            fraction_resampled_goals_are_env_goals=0,
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
    )
    train_vae_variant = dict(
        generate_vae_data_fctn=generate_vae_dataset,
        beta=5,
        num_epochs=500,
        algo_kwargs=dict(
            batch_size=64,
        ),
        vae_kwargs=dict(
            min_variance=None,
            input_channels=3,
        ),
        generate_vae_dataset_kwargs=dict(
            N=2500,
            use_cached=False,
            action_plus_random_sampling=True,
        ),
        representation_size=6,
        save_period=50,
    )
    variant = dict(
        grill_variant=grill_variant,
        train_vae_variant=train_vae_variant,
        env_class=SawyerDoorPushOpenActionLimitedEnv,
        env_kwargs=dict(
        ),
        init_camera=sawyer_door_env_camera_closer,
    )

    search_space = {
        'grill_variant.training_mode': ['train_env_goals'],
        # 'grill_variant.replay_kwargs.fraction_goals_are_rollout_goals': [0, 1],
        # 'grill_variant.replay_kwargs.fraction_resampled_goals_are_env_goals': [0, 1],
       }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_td3_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=5,
            )
