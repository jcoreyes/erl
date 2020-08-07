import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import (
    init_sawyer_camera_v4,
    init_sawyer_camera_v5
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEnv
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import train_vae

if __name__ == "__main__":
    variant = dict(
        representation_size=64,
        beta=1.0,
        num_epochs=500,
        generate_vae_dataset_kwargs=dict(
            # dataset_path='manual-upload/SawyerPushAndReachXYEnv_1000_init_sawyer_camera_v5_oracleTrue.npy',
            dataset_path='manual-upload/SawyerPushAndReachXYEnv_1000_init_sawyer_camera_v4_oracleTrue.npy',
            env_class=SawyerPushAndReachXYEnv,
            env_kwargs=dict(
                hide_goal_markers=True,
                puck_low=(-0.15, 0.5),
                puck_high=(0.15, 0.7),
                hand_low=(-0.2, 0.5, 0.),
                hand_high=(0.2, 0.7, 0.5),
            ),
            init_camera=init_sawyer_camera_v5,
            N=1000,
            oracle_dataset=True,
            num_channels=3,
            # show=True,
            # use_cached=False,
        ),
        algo_kwargs=dict(
            do_scatterplot=False,
            lr=1e-3,
        ),
        vae_kwargs=dict(
            input_channels=3,
        ),
        beta_schedule_kwargs=dict(
            x_values=[0, 100, 200, 500],
            y_values=[0, 0, 5, 5],
        ),
        save_period=10,
    )

    search_space = {
        'representation_size': [16],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    mode = 'local'
    exp_prefix = 'dev'

    # mode = 'ec2'
    exp_prefix = 'train-vae-beta-5-push-and-reach-cam4-y15-range'
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
            num_exps_per_instance=3,
        )
