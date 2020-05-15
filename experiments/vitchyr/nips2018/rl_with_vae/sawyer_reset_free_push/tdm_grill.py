import railrl.misc.hyperparameter as hyp
from railrl.envs.mujoco.sawyer_reset_free_push_env import SawyerResetFreePushEnv
from railrl.images.camera import sawyer_init_camera_zoomed_in_fixed
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.networks.experimental import HuberLoss
from railrl.torch.vae.tdm_td3_vae_experiment import tdm_td3_vae_experiment

if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 2
    mode = 'ec2'
    exp_prefix = 'grill-tdm-td3-sawyer-push-reset-free-small-joint-limits-sweep-tau'

    vae_paths = {
        "16": "05-22-vae-sawyer-reset-free-zoomed-in/05-22-vae-sawyer-reset"
              "-free-zoomed-in_2018_05_22_17_08_31_0000--s-51746-r16/params.pkl"
        # "16": "05-23-vae-sawyer-pusher-reset-free-large-joint-limts/05-23-vae-sawyer-pusher-reset-free-large-joint-limts_2018_05_23_16_30_36_0000--s-5828-r16/params.pkl",
    }

    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=101,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=1,
                reward_scale=1e-4,
                min_num_steps_before_training=1000,
            ),
            tdm_kwargs=dict(
                max_tau=15,
                num_pretrain_paths=0,
                reward_type='env',
            ),
            td3_kwargs=dict(
            ),
        ),
        env_kwargs=dict(
            hide_goal=True,
            # puck_limit='large',
            puck_limit='normal',
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
            structure='norm_difference',
        ),
        qf_criterion_class=HuberLoss,
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        replay_kwargs=dict(
            max_size=100000,
            fraction_goals_are_rollout_goals=0.2,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        algorithm='HER-TD3',
        normalize=False,
        rdim=32,
        render=False,
        env=SawyerResetFreePushEnv,
        use_env_goals=True,
        vae_paths=vae_paths,
        wrap_mujoco_env=True,
        do_state_based_exp=False,
        exploration_noise=0.1,
        init_camera=sawyer_init_camera_zoomed_in_fixed,
        version='normal',
        reward_params=dict(
            min_variance=0,
        ),
        es_kwargs=dict(
        ),
        tau_schedule_kwargs=dict(
            x_values=[0, 20, 40, 60, 80, 100],
        ),
    )

    search_space = {
        'exploration_type': [
            # 'epsilon',
            # 'gaussian',
            'ou',
        ],
        'algo_kwargs.base_kwargs.num_updates_per_env_step': [5],
        'tau_schedule_kwargs.y_values': [
            [0, 3, 6, 9, 12, 15],
            [15, 15, 15, 15, 15, 15],
            [0, 5, 10, 15, 25, 30],
            [30, 30, 30, 30, 30, 30],
            [0, 10, 20, 30, 40, 50],
            [50, 50, 50, 50, 50, 50],
        ],
        'replay_kwargs.fraction_resampled_goals_are_env_goals': [0.5],
        'replay_kwargs.fraction_goals_are_rollout_goals': [0.2],
        'exploration_noise': [0.2],
        'training_mode': ['train'],
        'testing_mode': ['test', ],
        # 'rdim': [2, 4, 8, 16],
        'rdim': [16],
        'reward_params.type': [
            'latent_distance',
            # 'log_prob',
            # 'mahalanobis_distance'
        ],
        'reward_params.min_variance': [0],
        'vae_wrapped_env_kwargs.sample_from_true_prior': [False],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                tdm_td3_vae_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                snapshot_mode='gap_and_last',
                snapshot_gap=10,
            )
