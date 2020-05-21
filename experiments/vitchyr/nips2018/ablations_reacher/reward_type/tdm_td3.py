from railrl.envs.mujoco.sawyer_gripper_env import SawyerXYEnv
from railrl.images.camera import sawyer_init_camera

import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.networks.experimental import HuberLoss
from railrl.torch.vae.tdm_td3_vae_experiment import tdm_td3_vae_experiment

if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 5
    mode = 'ec2'
    exp_prefix = 'reach-ablation-reward-type-2'

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
        "16": "05-12-sawyer-vae-reacher-no-min-var/05-12-sawyer-vae-reacher-no-min-var_2018_05_12_23_51_16_0000--s-15031-r16/params.pkl"
    }

    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=200,
                num_steps_per_epoch=50,
                num_steps_per_eval=1000,
                max_path_length=50,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=1,
                reward_scale=1e-4,
                min_num_steps_before_training=128,
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
            fraction_goals_are_rollout_goals=0.2,
            fraction_goals_are_env_goals=0.5,
        ),
        algorithm='TDM',
        normalize=False,
        rdim=32,
        render=False,
        env=SawyerXYEnv,
        use_env_goals=True,
        vae_paths=vae_paths,
        wrap_mujoco_env=True,
        do_state_based_exp=False,
        exploration_noise=0.1,
        init_camera=sawyer_init_camera,
        version='normal',
        reward_params=dict(
            min_variance=0,
        ),
        es_kwargs=dict(
        ),
    )

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.base_kwargs.num_updates_per_env_step': [5],
        'replay_kwargs.fraction_resampled_goals_are_env_goals': [0.5],
        'replay_kwargs.fraction_goals_are_rollout_goals': [0.2],
        'exploration_noise': [0.2],
        'training_mode': ['train'],
        'testing_mode': ['test', ],
        'rdim': [16],
        'reward_params.type': ['latent_distance'],
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
            )
