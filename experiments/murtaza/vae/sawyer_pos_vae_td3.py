from sawyer_control.sawyer_reaching import SawyerXYZReachingImgMultitaskEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants
import rlkit.misc.hyperparameter as hyp
from rlkit.torch.vae.relabeled_vae_experiment import experiment
import ray
ray.init()

if __name__ == "__main__":
    # noinspection PyTypeChecker
    vae_paths = {
        "16": "/home/mdalal/Documents/railrl-private/data/local/05-13-sawyer-vae-train/05-13-sawyer_vae_train_2018_05_13_12_47_34_0000--s-49137/itr_940.pkl",
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
            collection_mode='online-parallel'
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
        snapshot_mode='last',
        mode='here_no_doodad',
    )

    n_seeds = 1

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.num_updates_per_env_step': [5],
        'algo_kwargs.discount': [0.98],
        'replay_kwargs.fraction_goals_are_env_goals': [0, 0.5],
        'replay_kwargs.fraction_goals_are_rollout_goals': [0.2],
        'exploration_noise': [0.2],
        'algo_kwargs.reward_scale': [1e-4],
        'training_mode': ['train', ],
        'testing_mode': ['test', ],
        'rdim': [16],
        'seedid': range(n_seeds),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    run_variants(experiment, sweeper.iterate_hyperparameters(), run_id=10)
