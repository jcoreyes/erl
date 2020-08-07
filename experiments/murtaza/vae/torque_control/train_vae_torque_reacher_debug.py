from sklearn.model_selection import train_test_split

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
    #this has both states and images so can't use generate vae dataset
    X = np.load('/home/murtaza/vae_data/sawyer_torque_control_ou_imgs_zoomed_out10000.npy')
    Y = np.load('/home/murtaza/vae_data/sawyer_torque_control_ou_states_zoomed_out10000.npy')
    Y = np.concatenate((Y[:, :7], Y[:, 14:]), axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.1)
    info = dict()
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(**variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    m = ConvVAE(representation_size, input_channels=3, state_sim_debug=True, state_size = Y.shape[1], **variant['conv_vae_kwargs'])
    if ptu.gpu_enabled():
        m.cuda()
    t = ConvVAETrainer((X_train, Y_train), (X_test, Y_test), m, beta=beta,
                       beta_schedule=beta_schedule, state_sim_debug=True, **variant['algo_kwargs'])
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
    exp_prefix = 'sawyer_torque_vae_with_mse_loss_sweep'
    use_gpu = True

    variant = dict(
        beta=5,
        num_epochs=500,
        get_data_kwargs=dict(
            N=10000,
            use_cached=True,
        ),
        algo_kwargs=dict(
            mse_weight=.1,
        ),
        conv_vae_kwargs=dict(
            min_variance=None,
        ),
        save_period=1,
    )

    search_space = {
        'representation_size': [16, 32],
        'algo_kwargs.mse_weight':[10, 1, .1, .01],
        'beta':[4, 5, 6, 10]
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
