import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import (
    sawyer_pusher_camera_top_down,
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env_two_pucks import \
    SawyerPushAndReachXYDoublePuckEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import train_vae

if __name__ == "__main__":
    variant = dict(
        imsize=48,
        representation_size=64,
        beta=1.0,
        num_epochs=1000,
        generate_vae_dataset_kwargs=dict(
            env_class=SawyerPushAndReachXYDoublePuckEnv,
            env_kwargs=dict(
                hide_goal_markers=True,
                action_scale=.02,
            ),
            init_camera=sawyer_pusher_camera_top_down,
            N=1000,
            oracle_dataset=True,
            num_channels=3,
        ),
        algo_kwargs=dict(
            do_scatterplot=False,
            lr=1e-3,
        ),
        vae_kwargs=dict(
            input_channels=3,
        ),
        beta_schedule_kwargs=dict(
            x_values=[0, 100, 200, 1000],
            y_values=[0, 0, 5, 5],
        ),
        save_period=10,
    )

    search_space = {
        'representation_size': [32, 16],
        'generate_vae_dataset_kwargs.env_kwargs.always_start_on_same_side': [
            True, False],
        'generate_vae_dataset_kwargs.env_kwargs.goal_always_on_same_side': [
            True, False
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    mode = 'local'
    exp_prefix = 'dev'

    mode = 'ec2'
    exp_prefix = 'pre-train-two-puck-vae'
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
