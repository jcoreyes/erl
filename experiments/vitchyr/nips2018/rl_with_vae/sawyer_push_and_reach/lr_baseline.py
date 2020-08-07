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
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 5
    mode = 'ec2'
    exp_prefix = 'lr-baseline-pusher-0.2-range'

    vae_paths = {
        '16--lr-1e-3': "06-06-ae-sawyer-push/06-06-ae-sawyer"
                   "-push_2018_06_06_10_48_18_0000--s-55441-r16/params.pkl",
        '16--lr-1e-2': "06-06-ae-sawyer-push-lre-2/06-06-ae-sawyer-push-lre"
                       "-2_2018_06_06_14_12_13_0000--s-59123-r16/params.pkl",
    }

    variant = dict(
        algo_kwargs=dict(
            num_epochs=300,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            tau=1e-2,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
            # min_num_steps_before_training=1000,
        ),
        env_kwargs=dict(
            hide_goal=True,
            # reward_info=dict(
            #     type="shaped",
            # ),
        ),
        replay_kwargs=dict(
            fraction_goals_are_rollout_goals=0,
            fraction_resampled_goals_are_env_goals=1,
        ),
        algorithm='LR',
        normalize=False,
        rdim=16,
        render=False,
        env=SawyerPushAndReachXYEasyEnv,
        use_env_goals=True,
        vae_paths=vae_paths,
        wrap_mujoco_env=True,
        do_state_based_exp=False,
        exploration_noise=0.1,
        init_camera=sawyer_init_camera_zoomed_in,
        history_size=1,
    )

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.num_updates_per_env_step': [4],
        'exploration_noise': [0.2],
        'algo_kwargs.reward_scale': [1e-4],
        'reward_params.type': [
            'latent_distance',
        ],
        'training_mode': ['train'],
        'testing_mode': ['test', ],
        'init_camera': [
            sawyer_init_camera_zoomed_in,
        ],
        'rdim': [
            '16--lr-1e-3',
            '16--lr-1e-2',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                # snapshot_gap=50,
                # snapshot_mode='gap_and_last',
            )
