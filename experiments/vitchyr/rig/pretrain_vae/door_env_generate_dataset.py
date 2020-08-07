import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import (
    sawyer_door_env_camera,
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import generate_vae_dataset

if __name__ == "__main__":
    variant = dict(
        imsize=48,
        env_class=SawyerDoorEnv,
        env_kwargs=dict(),
        init_camera=sawyer_door_env_camera,
        N=100,
        oracle_dataset=False,
        num_channels=3,
        save_file_prefix='door_open',
        n_random_steps=30,
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
    # exp_prefix = 'pre-train-door'
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        run_experiment(
            generate_vae_dataset,
            exp_prefix=exp_prefix,
            mode=mode,
            variant=variant,
            use_gpu=True,
            snapshot_mode='last',
            exp_id=exp_id,
            num_exps_per_instance=1,
        )
