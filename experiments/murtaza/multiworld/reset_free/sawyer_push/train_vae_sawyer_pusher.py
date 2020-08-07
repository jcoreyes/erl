import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import init_sawyer_camera_v3, sawyer_pusher_camera_upright, sawyer_door_env_camera, \
    sawyer_pusher_camera_top_down, sawyer_pusher_camera_upright_v2
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import SawyerPushAndReachXYEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.misc.ml_util import PiecewiseLinearSchedule
from rlkit.torch.vae.conv_vae import ConvVAE
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from rlkit.torch.grill.launcher import generate_vae_dataset

def experiment(variant):
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data, info = generate_vae_dataset(
        **variant['generate_vae_dataset_kwargs']
    )
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        # kwargs = variant['beta_schedule_kwargs']
        # kwargs['y_values'][2] = variant['beta']
        # kwargs['x_values'][1] = variant['flat_x']
        # kwargs['x_values'][2] = variant['ramp_x'] + variant['flat_x']
        variant['beta_schedule_kwargs']['y_values'][-1] = variant['beta']
        beta_schedule = PiecewiseLinearSchedule(**variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    m = ConvVAE(representation_size, input_channels=3, **variant['conv_vae_kwargs'])
    if ptu.gpu_enabled():
        m.cuda()
    t = ConvVAETrainer(train_data, test_data, m, beta=beta,
                       beta_schedule=beta_schedule, **variant['algo_kwargs'])
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(epoch, save_reconstruction=should_save_imgs,
                     save_scatterplot=should_save_imgs)
        if should_save_imgs:
            t.dump_samples(epoch)


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'sawyer_pusher_vae_arena_large_puck'

    # n_seeds = 1
    # mode = 'ec2'
    # exp_prefix = 'sawyer_pusher_vae_real_world_goal_space_large_puck'

    use_gpu = True

    variant = dict(
        num_epochs=5000,
        algo_kwargs=dict(
            is_auto_encoder=False,
            batch_size=64,
            lr=1e-3,
        ),
        generate_vae_dataset_kwargs=dict(
            N=20000,
            oracle_dataset=True,
            use_cached=True,
            env_class=SawyerPushAndReachXYEnv,
            env_kwargs=dict(
                hide_goal_markers=True,
                reward_type='puck_distance',
                hand_low=(-0.28, 0.3, 0.05),
                hand_high=(0.28, 0.9, 0.3),
                puck_low=(-.4, .2),
                puck_high=(.4, 1),
                goal_low=(-0.28, 0.3, 0.02, -.2, .4),
                goal_high=(0.28, 0.9, 0.02, .2, .8),
            ),
            init_camera=sawyer_pusher_camera_upright_v2,
            show=False,
            tag='arena'
        ),
        # beta_schedule_kwargs=dict(
        #     x_values=[0, 1000, 3000],
        #     y_values=[0, 0, 1],
        # ),
        conv_vae_kwargs=dict(),
        save_period=100,
        beta=5,
        representation_size=16,
    )

    search_space = {
        'beta':[2.5]
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
