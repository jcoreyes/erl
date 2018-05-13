from sawyer_control.sawyer_reaching import SawyerXYZReachingImgMultitaskEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.launchers.arglauncher import run_variants
import railrl.misc.hyperparameter as hyp
from railrl.torch.vae.relabeled_vae_experiment import experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    vae_paths = {
        "4": "/home/mdalal/Documents/railrl-private/data/local/05-12-sawyer-vae-train/05-12-sawyer_vae_train_2018_05_12_23_48_18_0000--s-89945/itr_800.pkl",
        # "32": "ashvin/vae/sawyer3d/run0/id1/itr_980.pkl",
        # "64": "ashvin/vae/sawyer3d/run0/id2/itr_980.pkl"
    }

    variant = dict(
        algo_kwargs=dict(
            num_epochs=20,
            num_steps_per_epoch=500,
            num_steps_per_eval=500,
            tau=1e-2,
            batch_size=128,
            max_path_length=50,
            discount=0.95,
            # qf_learning_rate=1e-3,
            # policy_learning_rate=1e-4,
        ),
        env_kwargs=dict(
            action_mode='position',
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
        snapshot_mode='gap',
        snapshot_gap=1,
        mode='here_no_doodad',
    )

    n_seeds = 1

    search_space = {
        'exploration_type': [
            'ou',
        ],
        #'env_kwargs.arm_range': [0.5],
        #'env_kwargs.reward_params.epsilon': [0.5],
        'algo_kwargs.num_updates_per_env_step': [5],
        'algo_kwargs.discount': [0.98],
        'replay_kwargs.fraction_goals_are_env_goals': [0, 0.5], # 0.0 is normal, 0.5 means half goals are resampled from env
        'replay_kwargs.fraction_goals_are_rollout_goals': [0.2],#[0.2, 1.0], # 1.0 is normal, 0.2 is (future, k=4) HER
        'exploration_noise': [0.2],
        'algo_kwargs.reward_scale': [1e-4], # use ~1e-4 for VAE experiments
        'training_mode': ['train', ],
        'testing_mode': ['test', ],
        'rdim': [4], # Sweep only for VAE experiments
        'seedid': range(n_seeds),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    run_variants(experiment, sweeper.iterate_hyperparameters(), run_id=10)
