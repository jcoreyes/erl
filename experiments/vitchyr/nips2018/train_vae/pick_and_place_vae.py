import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import init_sawyer_camera_v1, \
    init_sawyer_camera_v2
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import \
    SawyerPickAndPlaceEnv
from rlkit.envs.mujoco.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEasyEnv
)
from rlkit.images.camera import (
    sawyer_init_camera_zoomed_in_fixed,
    sawyer_init_camera_zoomed_in,
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import (
    SawyerPickAndPlaceEnv
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import (
    SawyerPushAndReachXYEnv
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import train_vae
from rlkit.torch.vae.sawyer2d_push_variable_data import generate_vae_dataset

if __name__ == "__main__":
    env_class = SawyerPickAndPlaceEnv
    init_camera = init_sawyer_camera_v1
    variant = dict(
        env_class=env_class,
        init_camera=init_camera,
        representation_size=16,
        beta=5.0,
        num_epochs=500,
        generate_vae_dataset_kwargs=dict(
            env_class=env_class,
            N=1000,
            oracle_dataset=True,
            init_camera=init_camera,
            # show=True,
            # use_cached=False,
        ),
        algo_kwargs=dict(
            do_scatterplot=False,
            lr=1e-3,
        ),
        beta_schedule_kwargs=dict(
            x_values=[0, 100, 200, 500],
            y_values=[0, 0, 5, 5],
        ),
        save_period=5,
    )

    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    # n_seeds = 3
    # mode = 'ec2'
    # exp_prefix = 'multiworld-train-pnp-vae'
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                train_vae,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                snapshot_gap=50,
                snapshot_mode='gap_and_last',
            )
