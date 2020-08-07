import rlkit.misc.hyperparameter as hyp
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorPullOpenActionLimitedEnv
from multiworld.envs.mujoco.cameras import (
    sawyer_door_env_camera_closer)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_full_experiment
from rlkit.torch.vae.sawyer_door_pull_open_data import generate_vae_dataset

if __name__ == "__main__":
#    n_seeds = 1
#    mode = 'local'
#    exp_prefix = 'test'

    n_seeds = 5
    mode = 'local'
    exp_prefix = 'sawyer-door-pull-open-mw-grill-her-td3-final-revert-again'

    grill_variant = dict(
        do_state_exp=True,
        algo_kwargs=dict(
            num_epochs=1000,
            num_steps_per_epoch=2000,
            num_steps_per_eval=2000,
            batch_size=128,
            max_path_length=200,
            discount=0.99,
            min_num_steps_before_training=1000,
            reward_scale=100,
            collection_mode='online-parallel',
            sim_throttle=True,
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
        save_video=False,
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
        num_epochs=0,
        algo_kwargs=dict(
            batch_size=64,
        ),
        vae_kwargs=dict(
            min_variance=None,
            input_channels=3,
        ),
        generate_vae_dataset_kwargs=dict(
            N=50,
            use_cached=True,
            show=False,
            action_plus_random_sampling=True,
            ratio_action_sample_to_random=1,
        ),
        representation_size=16,
        save_period=50,
    )
    variant = dict(
        grill_variant=grill_variant,
        train_vae_variant=train_vae_variant,
        env_class=SawyerDoorPullOpenActionLimitedEnv,
        env_kwargs=dict(
            pos_action_scale=.02,
            min_y_pos=.5,
            max_y_pos=.6,
            use_line=True,
        ),
        init_camera=sawyer_door_env_camera_closer,
    )

    search_space = {
    
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                grill_her_td3_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=2,
            )
