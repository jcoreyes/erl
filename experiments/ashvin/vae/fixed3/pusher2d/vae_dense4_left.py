from rlkit.envs.multitask.point2d import MultitaskImagePoint2DEnv
from rlkit.envs.multitask.pusher2d import FullPusher2DEnv

from rlkit.launchers.arglauncher import run_variants
import rlkit.misc.hyperparameter as hyp
from rlkit.torch.vae.relabeled_vae_experiment import experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    vae_paths = {
        "4": "ashvin/vae/new-pusher2d/run12/id0/itr_1000.pkl",
        "8": "ashvin/vae/new-pusher2d/run12/id1/itr_1000.pkl",
        "16": "ashvin/vae/new-pusher2d/run12/id2/itr_1000.pkl",
        "32": "ashvin/vae/new-pusher2d/run12/id3/itr_1000.pkl"
    }

    variant = dict(
        algo_kwargs=dict(
            num_epochs=505,
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
            include_puck=True,
            arm_range=0.1,
            use_big_red_puck=True,
            reward_params=dict(
                type="sparse",
                epsilon=0.2,
                # puck_reward_only=True,
            ),
            only_far_goals=True,
        ),
        replay_kwargs=dict(
            fraction_goals_are_rollout_goals=0.2,
            fraction_goals_are_env_goals=0.5,
        ),
        algorithm='TD3',
        normalize=False,
        rdim=4,
        render=False,
        env=FullPusher2DEnv,
        use_env_goals=True,
        vae_paths=vae_paths,
        wrap_mujoco_env=True,
        do_state_based_exp=False,
        exploration_noise=0.1,
    )

    n_seeds = 3

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'env_kwargs.arm_range': [0.5],
        'env_kwargs.reward_params.epsilon': [0.5],
        'algo_kwargs.num_updates_per_env_step': [1, 4],
        'algo_kwargs.batch_size': [512],
        'algo_kwargs.discount': [0.99],
        'replay_kwargs.fraction_goals_are_env_goals': [0.5,],
        'replay_kwargs.fraction_goals_are_rollout_goals': [0.2,],
        'exploration_noise': [0.5],
        'algo_kwargs.reward_scale': [1e-4],
        'training_mode': ['train', ],
        'testing_mode': ['test', ],
        'rdim': [4, 8, 16, 32],
        'seedid': range(n_seeds),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    run_variants(experiment, sweeper.iterate_hyperparameters(), run_id=3)
