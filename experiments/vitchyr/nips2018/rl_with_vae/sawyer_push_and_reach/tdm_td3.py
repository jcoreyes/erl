import rlkit.misc.hyperparameter as hyp
from rlkit.envs.mujoco.sawyer_push_and_reach_env import \
    SawyerPushAndReachXYEasyEnv
from rlkit.images.camera import sawyer_init_camera_zoomed_in_fixed
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks.experimental import HuberLoss
from rlkit.torch.vae.tdm_td3_vae_experiment import tdm_td3_vae_experiment

if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 2
    mode = 'ec2'
    exp_prefix = 'grill-tdm-td3-push-range-0.2'

    vae_paths = {
        # "2": "05-11-sawyer-vae-reacher-recreate-results/05-11-sawyer-vae"
        #      "-reacher-recreate-results_2018_05_11_01_18_09_0000--s-33239-r2"
        #      "/params.pkl",
        # "4": "05-11-sawyer-vae-reacher-recreate-results/05-11-sawyer-vae"
        #      "-reacher-recreate-results_2018_05_11_01_21_47_0000--s-74741-r4"
        #      "/params.pkl",
        # "8": "05-11-sawyer-vae-reacher-recreate-results/05-11-sawyer-vae"
        #      "-reacher-recreate-results_2018_05_11_01_25_22_0000--s-82322-r8"
        #      "/params.pkl",
        # "16": "05-11-sawyer-vae-reacher-recreate-results/05-11-sawyer-vae"
        #       "-reacher-recreate-results_2018_05_11_01_28_52_0000--s-570-r16"
        #       "/params.pkl",
        "16": "05-23-vae-sawyer-variable-fixed-2/05-23-vae-sawyer-variable"
              "-fixed-2_2018_05_23_16_19_33_0000--s-293-nImg-1000--cam-sawyer_init_camera_zoomed_in_fixed/params.pkl",
    }

    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=250,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=16,
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
            # reward_info=dict(
            #     type="shaped",
            # ),
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
        algorithm='GRiLL-TDM-TD3',
        normalize=False,
        rdim=32,
        render=False,
        env=SawyerPushAndReachXYEasyEnv,
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
    )

    search_space = {
        'exploration_type': [
            # 'epsilon',
            # 'gaussian',
            'ou',
        ],
        'algo_kwargs.base_kwargs.num_updates_per_env_step': [4],
        'algo_kwargs.td3_kwargs.tau': [0.001, 0.05, 1],
        'algo_kwargs.tdm_kwargs.max_tau': [5, 50, 99],
        'replay_kwargs.fraction_resampled_goals_are_env_goals': [0.5],
        'replay_kwargs.fraction_goals_are_rollout_goals': [0.2],
        'algo_kwargs.tdm_kwargs.vectorized': [False],
        'qf_kwargs.structure': ['none'],
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
        if (
                variant['replay_kwargs']['fraction_goals_are_rollout_goals'] == 1.0
                and variant['replay_kwargs']['fraction_resampled_goals_are_env_goals'] == 0.5
        ):
            # redundant setting
            continue
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
