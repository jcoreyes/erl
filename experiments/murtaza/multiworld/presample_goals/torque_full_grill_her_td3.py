import rlkit.misc.hyperparameter as hyp
from rlkit.torch.vae.dataset.generate_goal_dataset import generate_goal_data_set
from multiworld.envs.mujoco.cameras import sawyer_torque_reacher_camera
from multiworld.envs.mujoco.sawyer_reach_torque.sawyer_reach_torque_env import SawyerReachTorqueEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_full_experiment
from rlkit.torch.vae.dataset.sawyer_torque_control_data import generate_vae_dataset

if __name__ == "__main__":
    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer_torque_multiworld_her_td3_grill_presampled_goals'

    grill_variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=201,
                num_steps_per_epoch=100,
                num_steps_per_eval=500,
                max_path_length=50,
                discount=0.99,
                batch_size=128,
                num_updates_per_env_step=1,
                reward_scale=100,
            ),
            td3_kwargs=dict(),
            her_kwargs=dict(),
        ),
        replay_buffer_kwargs=dict(
            fraction_goals_are_rollout_goals=0.5,
            fraction_resampled_goals_are_env_goals=0,
            max_size=int(1e6)
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        algorithm='GRILL-HER-TD3',
        normalize=False,
        render=False,
        use_env_goals=True,
        exploration_noise=0.3,
        version='normal',
        training_mode='train',
        testing_mode='test',
        reward_params=dict(
            min_variance=0,
            type='latent_distance',
        ),
        exploration_type='ou',
        vae_wrapped_env_kwargs=dict(
        ),
        generate_goal_dataset_fn=generate_goal_data_set,
        goal_generation_kwargs=dict(num_goals=1000, use_cached_dataset=False,),
        presample_goals=True,
        save_video_period=50,
        save_video=True,
      )
    train_vae_variant = dict(
        generate_vae_data_fctn=generate_vae_dataset,
        beta=1,
        num_epochs=1000,
        generate_vae_dataset_kwargs=dict(
            N=20000,
            use_cached=True ,
        ),
        algo_kwargs=dict(
            batch_size=64,
        ),
        vae_kwargs=dict(
            min_variance=None,
            input_channels=3,
        ),
        save_period=200,
        representation_size=16,
    )
    variant = dict(
        grill_variant=grill_variant,
        train_vae_variant=train_vae_variant,
        env_class=SawyerReachTorqueEnv,
        env_kwargs=dict(
            keep_vel_in_obs=True,
            use_safety_box=True,
        ),
        init_camera=sawyer_torque_reacher_camera,
    )
    search_space = {
        'grill_variant.algo_kwargs.base_kwargs.collection_mode':['online', 'online-parallel']
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
                num_exps_per_instance=3,
            )
