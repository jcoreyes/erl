import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.misc.ml_util import PiecewiseLinearSchedule
from rlkit.torch.vae.conv_vae import ConvVAE
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
import numpy as np

def experiment(variant):
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    # train_data, test_data, info = generate_vae_dataset(
    #     **variant['get_data_kwargs']
    # )
    num_divisions = 5
    images = np.zeros((num_divisions * 10000, 21168))
    for i in range(num_divisions):
        imgs = np.load('/home/murtaza/vae_data/sawyer_torque_control_images100000_' + str(i + 1) + '.npy')
        images[i * 10000:(i + 1) * 10000] = imgs
        print(i)
    mid = int(num_divisions * 10000 * .9)
    train_data, test_data = images[:mid], images[mid:]
    info = dict()

    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        kwargs = variant['beta_schedule_kwargs']
        kwargs['y_values'][2] = variant['beta']
        kwargs['x_values'][1] = variant['flat_x']
        kwargs['x_values'][2] = variant['ramp_x'] + variant['flat_x']
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
    exp_prefix = 'sawyer_torque_vae'
    use_gpu = True

    variant = dict(
        beta=5,
        num_epochs=100,
        algo_kwargs=dict(
            batch_size=256,
        ),
        conv_vae_kwargs=dict(
            min_variance=None,
        ),
        save_period=10,
    )

    search_space = {
        'representation_size': [32],
        'algo_kwargs.lr':[1e-3],
        'beta':[5],
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
                snapshot_mode='gap',
                snapshot_gap=20,
            )
