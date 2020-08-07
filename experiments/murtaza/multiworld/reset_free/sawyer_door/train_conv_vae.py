import rlkit.misc.hyperparameter as hyp
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
    m = variant['vae'](representation_size, **variant['vae_kwargs'])
    m.to(ptu.device)
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
            if variant['dump_skew_debug_plots']:
                t.dump_best_reconstruction(epoch)
                t.dump_worst_reconstruction(epoch)
                t.dump_sampling_histogram(epoch)
        t.update_train_weights()


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'test'

    # n_seeds = 1
    # mode = 'ec2'
    # exp_prefix = 'normalized-sampling'

    use_gpu = True

    architecture = dict(
        conv_args=dict(
            kernel_sizes=[5, 3, 3],
            n_channels=[16, 32, 64],
            strides=[3, 2, 2],
        ),
        conv_kwargs=dict(
            hidden_sizes=[500, 300, 150],
        ),
        deconv_args=dict(
            hidden_sizes=[150, 300, 500],

            deconv_input_width=3,
            deconv_input_height=3,
            deconv_input_channels=64,

            deconv_output_kernel_size=6,
            deconv_output_strides=3,
            deconv_output_channels=3,

            kernel_sizes=[3, 3],
            n_channels=[32, 16],
            strides=[2, 2],
        ),
        deconv_kwargs=dict(
        )
    )

    variant = dict(
        num_epochs=1000,
        algo_kwargs=dict(
            is_auto_encoder=False,
            batch_size=64,
            lr=1e-3,
            skew_config=dict(
                method='inv_gaussian_p_x',
            ),
            skew_dataset=False,
        ),
        vae=ConvVAE,
        dump_skew_debug_plots=True,
        generate_vae_dataset_fn=generate_vae_dataset,
        generate_vae_dataset_kwargs=dict(
            N=5000,
            dataset_path='datasets/SawyerDoorHookResetFreeEnv-v5_N5000_sawyer_door_env_camera_v3_imsize48_random_oracle_split_0.9_twin_sac.npy',
            oracle_dataset=False,
            use_cached=True,
            oracle_dataset_from_policy=True,
            imsize=48,
            non_presampled_goal_img_is_garbage=True,
            vae_dataset_specific_kwargs=dict(),
            policy_file='09-22-sawyer-door-new-door-60-reset-free-space-fix/09-22-sawyer_door_new_door_60_reset_free_space_fix_2018_09_23_04_05_41_id000--s34898/params.pkl',
            n_random_steps=100,
            show=False,
        ),
        vae_kwargs=dict(
            input_channels=3,
            imsize=48,
            architecture=architecture,
            decoder_distribution='gaussian_identity_variance'
        ),
        save_period=10,
        beta=.5,
        representation_size=16,
    )

    search_space = {
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
                snapshot_gap=100,
            )
