import railrl.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v3
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_hook import SawyerDoorHookEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.ml_util import PiecewiseLinearSchedule
from railrl.torch.vae.conv_vae import ConvVAE, ConvVAETrainer
from railrl.torch.grill.launcher import generate_vae_dataset

def experiment(variant):
    from railrl.core import logger
    import railrl.torch.pytorch_util as ptu
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data, info = variant['generate_vae_dataset_fn'](
        variant['generate_vae_dataset_kwargs']
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
    m = ConvVAE(representation_size, **variant['vae_kwargs'])
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
    exp_prefix = 'sawyer_hook_door_vae'

    # n_seeds = 1
    # mode = 'ec2'
    # exp_prefix = 'sawyer_hook_door_vae'

    use_gpu = True

    variant = dict(
        num_epochs=5000,
        algo_kwargs=dict(
            is_auto_encoder=False,
            batch_size=64,
            lr=1e-3,
        ),
        generate_vae_dataset_fn=generate_vae_dataset,
        generate_vae_dataset_kwargs=dict(
            N=5000,
            oracle_dataset=False,
            use_cached=True,
            oracle_dataset_from_policy=True,
            imsize=48,
            env_class=SawyerDoorHookEnv,
            env_kwargs=dict(
                goal_low=(-0.1, 0.42, 0.05, 0),
                goal_high=(0.0, 0.65, .075, 1.0472),
                hand_low=(-0.1, 0.42, 0.05),
                hand_high=(0., 0.65, .075),
                max_angle=1.0472,
                xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            ),
            non_presampled_goal_img_is_garbage=True,
            vae_dataset_specific_kwargs=dict(),
            policy_file='09-22-sawyer-door-60-reset-free-thicker-handle/09-22-sawyer_door_60_reset_free_thicker_handle_2018_09_22_23_22_27_id000--s25274/params.pkl',
            n_random_steps=100,
            init_camera=sawyer_door_env_camera_v3,
            show=True,
        ),
        vae_kwargs=dict(
            input_channels=3,
        ),
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
