from rlkit.envs.mujoco.sawyer_push_and_reach_env import \
    SawyerPushAndReachXYEnv, SawyerPushAndReachXYEasyEnv
from rlkit.envs.mujoco.sawyer_push_env import SawyerPushXYEnv
from rlkit.envs.multitask.point2d import MultitaskImagePoint2DEnv
from rlkit.envs.multitask.pusher2d import FullPusher2DEnv
from rlkit.images.camera import sawyer_init_camera, \
    sawyer_init_camera_zoomed_in

from rlkit.launchers.arglauncher import run_variants
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.vae.relabeled_vae_experiment import experiment

if __name__ == "__main__":
    vae_paths = {
        "4": "ashvin/vae/fixed3/sawyer-pusher/train-vae-wider/run5/id0/itr_480.pkl",
        "16": "ashvin/vae/fixed3/sawyer-pusher/train-vae-wider/run5/id1/itr_480.pkl",
    }

    variant = dict(
        algo_kwargs=dict(
            num_epochs=105,
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
            hide_goal=True,
            # reward_info=dict(
            #     type="shaped",
            # ),
        ),
        replay_kwargs=dict(
            fraction_goals_are_rollout_goals=0.2,
            fraction_goals_are_env_goals=0.5,
        ),
        vae_wrapped_env_kwargs=dict(
            sample_from_true_prior=False,
            reward_params=dict(
                type="log_prob",
            )
        ),
        algorithm='HER-TD3',
        normalize=False,
        rdim=4,
        render=False,
        env=SawyerPushAndReachXYEasyEnv,
        use_env_goals=True,
        vae_paths=vae_paths,
        wrap_mujoco_env=True,
        do_state_based_exp=True,
        exploration_noise=0.1,
        init_camera=sawyer_init_camera_zoomed_in,
    )

    n_seeds = 3
    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.num_updates_per_env_step': [1, 4],
        'replay_kwargs.fraction_goals_are_env_goals': [0.0, 0.5, ],
        'replay_kwargs.fraction_goals_are_rollout_goals': [0.2, 1.0 ],
        'vae_wrapped_env_kwargs.sample_from_true_prior': [False],
        'exploration_noise': [0.2, ],
        'algo_kwargs.reward_scale': [1e-4],
        'training_mode': ['train'],
        'testing_mode': ['test', ],
        'rdim': [4],
        'seedid': range(n_seeds),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    run_variants(experiment, sweeper.iterate_hyperparameters(), run_id=1)
