
import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.ml_util import LinearSchedule
from railrl.torch.vae.conv_vae import ConvVAE, ConvVAETrainer
from railrl.torch.vae.sawyer2d_push_data import get_data


def experiment(variant):
    from railrl.core import logger
    import railrl.torch.pytorch_util as ptu
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data, info = get_data(**variant['get_data_kwargs'])
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    beta_schedule = LinearSchedule(
        **variant['beta_schedule_kwargs'],
    )
    m = ConvVAE(representation_size, input_channels=3)
    if ptu.gpu_enabled():
        m.cuda()
    t = ConvVAETrainer(train_data, test_data, m, beta=beta,
                       beta_schedule=beta_schedule)
    for epoch in range(variant['num_epochs']):
        t.train_epoch(epoch)
        t.test_epoch(epoch)
        t.dump_samples(epoch)


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-sawyer-push-vae'
    use_gpu = True

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer-push-vae-with-moving-arm-gpu'
    # use_gpu = False

    variant = dict(
        beta=5.0,
        num_epochs=100,
        get_data_kwargs=dict(
            N=5000,
        ),
        beta_schedule_kwargs=dict(
            init_value=0,
            final_value=5,
            ramp_duration=100,
        )
    )

    search_space = {
        'representation_size': [2, 4, 8, 16],
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
            )
