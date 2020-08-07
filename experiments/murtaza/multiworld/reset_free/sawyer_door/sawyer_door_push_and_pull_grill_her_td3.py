import rlkit.misc.hyperparameter as hyp
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_full_experiment
from rlkit.torch.vae.sawyer_door_push_and_pull_open_data import generate_vae_dataset

if __name__ == "__main__":
    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer_door_push_and_pull_open_mw_grill_her_td3_reset'

    grill_variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=5000,
                num_steps_per_epoch=1000,
                num_steps_per_eval=2000,
                batch_size=128,
                max_path_length=200,
                discount=0.99,
                min_num_steps_before_training=128,
                reward_scale=1,
                num_updates_per_env_step=1,
                collection_mode='online-parallel'
            ),
            her_kwargs=dict(),
            td3_kwargs=dict(),
        ),
        replay_buffer_class=ObsDictRelabelingBuffer,
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=.5,
            fraction_resampled_goals_are_env_goals=0,
        ),
        qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
         policy_kwargs=dict(
               hidden_sizes=[400, 300],
        ),
        algorithm='VAE-HER-TD3',
        normalize=False,
        render=False,
        version='normal',
        save_video=True,
        exploration_type='ou',
        exploration_noise=.8,
        training_mode='train',
        testing_mode='test',
        reward_params=dict(
            min_variance=0,
            type='latent_distance',
        ),
        save_video_period=200,
    )
    train_vae_variant = dict(
        beta=5,
        num_epochs=1,
        algo_kwargs=dict(
            batch_size=64,
        ),
        vae_kwargs=dict(
            min_variance=None,
            input_channels=3,
        ),
        generate_vae_dataset_kwargs=dict(
            N=100,
            use_cached=False,
            show=False,
            oracle_dataset=True,
        ),
        representation_size=16,
        save_period=200,
    )

    variant = dict(
        grill_variant=grill_variant,
        train_vae_variant=train_vae_variant,
        imsize=48,
    )

    search_space = {
        'env_id':['Image48SawyerPushAndPullDoorEnv-v0', 'Image48SawyerPushAndPullDoorEnvResetFree-v0'],

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
                num_exps_per_instance=3,
            )
