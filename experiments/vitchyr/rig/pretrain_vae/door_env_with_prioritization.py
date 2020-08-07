import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import (
    sawyer_pusher_camera_top_down,
    sawyer_door_env_camera,
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env_two_pucks import \
    SawyerPushAndReachXYDoublePuckEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import train_vae

if __name__ == "__main__":
    variant = dict(
        imsize=48,
        representation_size=16,
        beta=1.0,
        num_epochs=1000,
        generate_vae_dataset_kwargs=dict(
            dataset_path='/home/vitchyr/git/railrl/data/doodads3/manual'
                         '-upload/full_skewed_dataset_SawyerDoorEnv_N1000_sawyer_door_env_camera_imsize48_oracleFalse.npy',
        ),
        algo_kwargs=dict(
            do_scatterplot=False,
            lr=1e-3,
            batch_size=128,
        ),
        vae_kwargs=dict(
            input_channels=3,
        ),
        beta_schedule_kwargs=dict(
            x_values=[0, 100, 200, 1000],
            y_values=[0, 0, 5, 5],
        ),
        save_period=20,
    )

    search_space = {
        'algo_kwargs.skew_dataset': [
            False,
            # True,
        ],
        'algo_kwargs.skew_config.power': [
            # 2,
            1,
        ],
        'algo_kwargs.skew_config.method': [
            'kl',
            # 'squared_error',
        ],
        'algo_kwargs.gaussian_decoder_loss': [
            True,
            # False,
        ],
        'representation_size': [16],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    mode = 'local'
    exp_prefix = 'dev'

    # mode = 'ec2'
    exp_prefix = 'pre-train-door-on-skewed-dataset-skew-no-skew-training'
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        run_experiment(
            train_vae,
            exp_prefix=exp_prefix,
            mode=mode,
            variant=variant,
            use_gpu=True,
            snapshot_gap=50,
            snapshot_mode='gap_and_last',
            exp_id=exp_id,
            num_exps_per_instance=2,
        )
