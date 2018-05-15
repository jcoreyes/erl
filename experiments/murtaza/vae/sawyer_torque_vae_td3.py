from sawyer_control.sawyer_reaching import SawyerXYZReachingImgMultitaskEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.launchers.arglauncher import run_variants
import railrl.misc.hyperparameter as hyp
from railrl.torch.vae.relabeled_vae_experiment import experiment

if __name__ == "__main__":
    vae_paths = {
        "16": "/home/mdalal/Documents/railrl-private/data/local/05-13-sawyer-vae-train/05-13-sawyer_vae_train_2018_05_13_12_47_34_0000--s-49137/itr_1000.pkl",
        "32": "/home/mdalal/Documents/railrl-private/data/local/05-13-sawyer-vae-train/05-13-sawyer_vae_train_2018_05_13_12_48_26_0000--s-87652/itr_1000.pkl",
        "64": "/home/mdalal/Documents/railrl-private/data/local/05-13-sawyer-vae-train/05-13-sawyer_vae_train_2018_05_13_12_49_07_0000--s-13504/itr_1000.pkl",

    }

    variant = dict(
        algo_kwargs=dict(
            num_epochs=100,
            num_steps_per_epoch=500,
            num_steps_per_eval=500,
            tau=1e-2,
            batch_size=512,
            max_path_length=100,
            discount=0.95,
        ),
        env_kwargs=dict(
            action_mode='torque',
            reward='norm'

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
    )

    n_seeds = 1

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.num_updates_per_env_step': [1, 3],
        'algo_kwargs.discount': [0.98],
        'replay_kwargs.fraction_goals_are_env_goals': [0.5], # 0.0 is normal, 0.5 means half goals are resampled from env
        'replay_kwargs.fraction_goals_are_rollout_goals': [0.2],#[0.2, 1.0], # 1.0 is normal, 0.2 is (future, k=4) HER
        'exploration_noise': [0.2],
        'algo_kwargs.reward_scale': [1e-4], # use ~1e-4 for VAE experiments
        'training_mode': ['train', ],
        'testing_mode': ['test', ],
        'rdim': [16, 32, 64], # Sweep only for VAE experiments
        'seedid': range(n_seeds),
    }
    # run_variants(experiment, sweeper.iterate_hyperparameters(), run_id=10)
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    for variant in sweeper.iterate_hyperparameters():
        n_seeds = 1
        exp_prefix = 'sawyer_torque_vae_td3_history'
        mode = 'here_no_doodad'
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                variant=variant,
            )