from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import generate_vae_dataset
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright
from rlkit.misc.ml_util import PiecewiseLinearSchedule
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env_reset import SawyerPushAndReachXYEnv
from rlkit.torch.vae.acai import ACAI, ACAITrainer
import rlkit.misc.hyperparameter as hyp

def experiment(variant):
    lmbda = variant['lmbda']
    gamma = variant['gamma']
    mu = variant['mu']

    beta = PiecewiseLinearSchedule([0, 2500, 3500], [0, 0, variant['beta']])
    representation_size = variant["representation_size"]
    train_data, test_data, info = generate_vae_dataset(
        **variant['generate_vae_dataset_kwargs']
    )
    m = ACAI(representation_size, input_channels=3)
    t = ACAITrainer(train_data, test_data, m, beta_schedule=beta, gamma=gamma, mu=mu, lmbda=lmbda)
    for epoch in range(6001):
        t.train_epoch(epoch)
        t.test_epoch(epoch)
        if epoch % variant['save_period'] == 0:
            t.dump_samples(epoch)

if __name__ == "__main__":
    use_gpu=True
    variant=dict(
        generate_vae_dataset_kwargs=dict(
            N=25000,
            oracle_dataset=True,
            use_cached=True,
            env_class=SawyerPushAndReachXYEnv,
            env_kwargs=dict(
                hide_goal_markers=True,
                reward_type='puck_distance',
                hand_low=(-0.275, 0.275, .0),
                hand_high=(0.275, 0.825, .5),
                puck_low=(-0.25, 0.3),
                puck_high=(0.25, 0.8),
                goal_low=(-0.25, 0.3),
                goal_high=(0.25, 0.8),
            ),
            init_camera=sawyer_pusher_camera_upright,
            show=False,
        ),
        lmbda=.5,
        gamma=.2,
        mu=1,
        representation_size=16,
        save_period=50,
        beta=.5,
    )

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer_pusher_acai_big_sweep'
    search_space = {
        'representation_size':[16, 32],
        'beta':[.25, .5, 1]

    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for _ in range(n_seeds):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=use_gpu,
                num_exps_per_instance=1,
                snapshot_mode='gap_and_last',
                snapshot_gap=500,
            )
