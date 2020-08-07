from sawyer_control.sawyer_reaching import SawyerXYZReachingImgMultitaskEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants
import rlkit.misc.hyperparameter as hyp
from rlkit.torch.vae.relabeled_vae_experiment import experiment

if __name__ == "__main__":
    vae_paths = {
        "16": "/home/mdalal/Documents/railrl-private/data/local/05-14-sawyer-torque-vae-train-16/05-14-sawyer_torque_vae_train_16_2018_05_14_21_48_53_0000--s-32499/itr_1000.pkl",
        "32": "/home/mdalal/Documents/railrl-private/data/local/05-14-sawyer-torque-vae-train-32/05-14-sawyer_torque_vae_train_32_2018_05_14_21_49_34_0000--s-13212/itr_1000.pkl",
        "64": "/home/mdalal/Documents/railrl-private/data/local/05-14-sawyer-torque-vae-train-64/05-14-sawyer_torque_vae_train_64_2018_05_14_22_08_58_0000--s-19762/itr_1000.pkl",

    }
    use_gpu=True
    variant = dict(
        algo_kwargs=dict(
            num_epochs=50,
            num_steps_per_epoch=1000,
            num_steps_per_eval=500,
            tau=1e-2,
            batch_size=128,
            max_path_length=100,
            discount=0.95,
        ),
        env_kwargs=dict(
            action_mode='torque',
            reward='norm',
            update_hz=100,

        ),
        replay_kwargs=dict(
            fraction_goals_are_rollout_goals=0.2,
            fraction_goals_are_env_goals=0.5,
        ),
        algorithm='TD3',
        normalize=False,
        rdim=16,
        render=False,
        env=SawyerXYZReachingImgMultitaskEnv,
        use_env_goals=True,
        vae_paths=vae_paths,
        wrap_mujoco_env=False,
        do_state_based_exp=False,
        exploration_noise=0.1,
        snapshot_mode='last',
        mode='here_no_doodad',
        use_gpu=use_gpu,
    )

    n_seeds = 1

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.num_updates_per_env_step': [3],
        'algo_kwargs.discount': [0.98],
        'replay_kwargs.fraction_goals_are_env_goals': [0, 0.5], # 0.0 is normal, 0.5 means half goals are resampled from env
        'replay_kwargs.fraction_goals_are_rollout_goals': [0.2],#[0.2, 1.0], # 1.0 is normal, 0.2 is (future, k=4) HER
        'exploration_noise': [0.25],
        'algo_kwargs.reward_scale': [1e-4], # use ~1e-4 for VAE experiments
        'training_mode': ['train', ],
        'testing_mode': ['test', ],
        'rdim': [16, 32, 64], # Sweep only for VAE experiments
        'seedid': range(n_seeds),
        'hidden_sizes':[[100, 100]],
    }
    # run_variants(experiment, sweeper.iterate_hyperparameters(), run_id=10)
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    for variant in sweeper.iterate_hyperparameters():
        n_seeds = 1
        exp_prefix = 'test'
        mode = 'here_no_doodad'
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                variant=variant,
                use_gpu=use_gpu,
            )