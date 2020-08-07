from rlkit.envs.multitask.point2d import MultitaskImagePoint2DEnv
from rlkit.envs.multitask.pusher2d import FullPusher2DEnv

from rlkit.launchers.arglauncher import run_variants
import rlkit.misc.hyperparameter as hyp
from rlkit.torch.vae.relabeled_vae_experiment import experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    vae_paths = {
        "2": "ashvin/vae/new-point2d/run3/id0/params.pkl",
        "4": "ashvin/vae/new-point2d/run3/id1/params.pkl",
        "8": "ashvin/vae/new-point2d/run3/id2/params.pkl",
        "16": "ashvin/vae/new-point2d/run3/id3/params.pkl"
    }

    variant = dict(
        algo_kwargs=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            tau=1e-2,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
            # qf_learning_rate=1e-3,
            # policy_learning_rate=1e-4,
        ),
        env_kwargs=dict(
            render_onscreen=False,
            render_size=84,
            ball_radius=1,
        ),
        replay_kwargs=dict(
            fraction_goals_are_rollout_goals=1.0,
            fraction_goals_are_env_goals=0.0,
        ),
        algorithm='TD3',
        normalize=False,
        rdim=4,
        render=False,
        env=MultitaskImagePoint2DEnv,
        use_env_goals=False,
        vae_paths=vae_paths,
        # save_video=False,
    )

    n_seeds = 3

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.reward_scale': [1e-4],
        'training_mode': ['train', 'test'],
        'testing_mode': ['test', ],
        'rdim': [2, 4, 8, 16],
        'seedid': range(n_seeds),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    run_variants(experiment, sweeper.iterate_hyperparameters(), run_id=3)
