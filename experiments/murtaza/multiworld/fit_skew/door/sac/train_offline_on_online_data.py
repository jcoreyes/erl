import joblib
from torch import nn
import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.ml_util import PiecewiseLinearSchedule
from railrl.torch.vae.conv_vae import imsize48_default_architecture, ConvVAE
from railrl.torch.vae.vae_trainer import ConvVAETrainer

def experiment(variant):
    from railrl.core import logger
    import railrl.torch.pytorch_util as ptu
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    data = joblib.load(variant['file'])
    obs = data['obs']
    size = data['size']
    dataset = obs[:size, :]
    n = int(size * .9)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    logger.get_snapshot_dir()
    print('SIZE: ', size)
    if variant.get('beta_schedule_kwargs', None):
        beta_schedule = PiecewiseLinearSchedule(**variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    m = variant['vae'](representation_size, decoder_output_activation=nn.Sigmoid(), **variant['vae_kwargs'])
    m.to(ptu.device)
    t = ConvVAETrainer(train_dataset, test_dataset, m, beta=beta,
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
        if epoch % 2 == 0:
            t.update_train_weights()


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'first10K_samples_fit_skew'

    # n_seeds = 1
    # mode = 'ec2'
    # exp_prefix = 'test'
    use_gpu=True

    variant = dict(
        file='/home/murtaza/research/railrl/data/local/11-15-test/11-15-test_2018_11_15_12_57_26_id000--s13644/extra_data.pkl',
        num_epochs=1000,
        algo_kwargs=dict(
            is_auto_encoder=False,
            batch_size=64,
            lr=1e-3,
            skew_config=dict(
                method='inv_bernoulli_p_x',
                power=1/2,
            ),
            skew_dataset=True,
        ),
        vae=ConvVAE,
        dump_skew_debug_plots=False,
        vae_kwargs=dict(
            input_channels=3,
            imsize=48,
            architecture=imsize48_default_architecture,
            decoder_distribution='bernoulli',
        ),
        save_period=10,
        beta=2.5,
        representation_size=16,
    )

    search_space = {
        'algo_kwargs.skew_config.power':[1/2, 1, 2],
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
                num_exps_per_instance=2,
                snapshot_mode='gap_and_last',
                snapshot_gap=100,
            )