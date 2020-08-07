from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import grill_her_td3_full_experiment
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place \
        import SawyerPickAndPlaceEnvYZ


if __name__ == "__main__":
    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer_pick_and_place_grill_reset'

    variant = dict(
        env_kwargs=dict(
            hide_arm=False,
            hide_goal_markers=True,
        ),

        env_class=SawyerPickAndPlaceEnvYZ,
        init_camera=sawyer_pick_and_place_camera,

        train_vae_variant = dict(
            beta=5.0,
            representation_size=8,
            generate_vae_dataset_kwargs=dict(
                N=7500,
                use_cached=False,
                imsize=84,
                num_channels=3,
                show=False,
                oracle_dataset=True,
            ),
            algo_kwargs=dict(
            ),
            beta_schedule_kwargs=dict(
                x_values=[0, 100, 200, 500, 1000],
                y_values=[0, 0, 0, 2.5, 5],
            ),
            save_period=200,
            num_epochs=1000,
        ),
        grill_variant = dict(
            algo_kwargs=dict(
                num_epochs=1000,
                num_steps_per_epoch=500,
                num_steps_per_eval=500,
                tau=1e-2,
                batch_size=128,
                max_path_length=100,
                discount=0.99,
                min_num_steps_before_training=1000,
            ),
            replay_buffer_kwargs=dict(
                max_size=int(1e6),
                fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0.5,
            ),
            algorithm='HER-TD3',
            normalize=False,
            render=False,
            use_env_goals=True,
            exploration_noise=0.3,
            save_video_period=200,
        )
    )

    search_space = {
        'grill_variant.exploration_type': [
            'ou'
        ],
        'grill_variant.algo_kwargs.num_updates_per_env_step': [1],
        'env_kwargs.oracle_reset_prob': [0.0, 0.5],
        'env_kwargs.action_scale': [.02],
        'grill_variant.exploration_noise': [0.3,.5],
        'grill_variant.algo_kwargs.reward_scale': [1, 100],
        'grill_variant.reward_params.type': [
            'latent_distance',
        ],
        'grill_variant.training_mode': ['train'],
        'grill_variant.testing_mode': ['test', ],
        'grill_variant.env_kwargs.reset_free':[True, False],
        'grill_variant.env_kwargs.random_init':[True, False],

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