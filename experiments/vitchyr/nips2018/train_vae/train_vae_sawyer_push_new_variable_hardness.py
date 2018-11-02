
import railrl.misc.hyperparameter as hyp
from railrl.images.camera import (
    # sawyer_init_camera_zoomed_in,
    # sawyer_init_camera,
    # sawyer_init_camera_zoomed_in_fixed,
    sawyer_init_camera_zoomed_out_fixed,
    sawyer_init_camera_zoomed_in_fixed)
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.ml_util import PiecewiseLinearSchedule
from railrl.torch.vae.conv_vae import ConvVAE
from railrl.torch.vae.vae_trainer import ConvVAETrainer
from railrl.torch.vae.sawyer2d_push_variable_data import generate_vae_dataset


def experiment(variant):
    from railrl.core import logger
    import railrl.torch.pytorch_util as ptu
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data, info = generate_vae_dataset(
        **variant['generate_vae_dataset_kwargs']
    )
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(**variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    m = ConvVAE(representation_size, input_channels=3)
    if ptu.gpu_enabled():
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


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-sawyer-push-new-vae'
    use_gpu = True

    # n_seeds = 1
    # mode = 'ec2'
    # exp_prefix = 'vae-sawyer-variable-zoomed-out-sweep'

    variant = dict(
        beta=5.0,
        num_epochs=500,
        generate_vae_dataset_kwargs=dict(
            N=1000,
            env_kwargs=dict(
                init_goal_low=(-0.15, 0.5),
                init_goal_high=(0.15, 0.7),
            ),
        ),
        algo_kwargs=dict(
            do_scatterplot=False,
            lr=1e-3,
        ),
        # TODO: automate this process
        beta_schedule_kwargs=dict(
            x_values=[0, 100, 200, 300, 400, 500],
            # y_values=[0, 0, 0.1, 0.5],
            y_values=[0, 0, 5, 5, 5, 5],
        ),
        save_period=10,
    )

    search_space = {
        'representation_size': [16],
        'beta_schedule_kwargs.y_values': [
            [0, 0, 0, 0.1, 0.5, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 5, 5],
            [0, 0, 0.5, 1, 1, 5],
        ],
        # 'algo_kwargs.lr': [1e-3, 1e-2],
        'generate_vae_dataset_kwargs.init_camera': [
            # sawyer_init_camera_zoomed_in,
            # sawyer_init_camera,
            # sawyer_init_camera_zoomed_in_fixed,
            sawyer_init_camera_zoomed_out_fixed,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for _ in range(n_seeds):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            suffix = "nImg-{}--cam-{}".format(
                variant['generate_vae_dataset_kwargs']['N'],
                variant['generate_vae_dataset_kwargs']['init_camera'].__name__,
            )
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=use_gpu,
                trial_dir_suffix=suffix,
            )
