from rlkit.envs.multitask.point2d import MultitaskImagePoint2DEnv
from rlkit.envs.multitask.pusher2d import FullPusher2DEnv

from rlkit.launchers.arglauncher import run_variants
import rlkit.misc.hyperparameter as hyp
from rlkit.torch.vae.relabeled_vae_experiment import experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    vae_paths = {
        "2": "ashvin/vae/new-pusher2d/run3/id0/params.pkl",
        "4": "ashvin/vae/new-pusher2d/run3/id1/params.pkl",
        "8": "ashvin/vae/new-pusher2d/run3/id2/params.pkl",
        "16": "ashvin/vae/new-pusher2d/run3/id3/params.pkl"
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
            ignore_multitask_goal=False,
            include_puck=True,
            arm_range=0.1,
            reward_params=dict(
                type="sparse",
                epsilon=0.1,
                puck_reward_only=True,
            ),
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
        track_qpos_goal=5,
        do_state_based_exp=True,
        exploration_noise=0.1,
    )

    n_seeds = 3

    search_space = {
        'exploration_type': [
            'epsilon',
        ],
        'env_kwargs.arm_range': [0.5],
        'algo_kwargs.reward_scale': [1],
        'algo_kwargs.num_updates_per_env_step': [1, 4, 16],
        'algo_kwargs.discount': [0.99],
        'exploration_noise': [0.2],
        'replay_kwargs.fraction_goals_are_env_goals': [0.0, 0.5,],
        'replay_kwargs.fraction_goals_are_rollout_goals': [0.2, 1.0],
        # 'rdim': [2, 4, 8, 16],
        'seedid': range(n_seeds),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    run_variants(experiment, sweeper.iterate_hyperparameters(), run_id=3)
