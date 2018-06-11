from railrl.core import logger
from railrl.misc.ml_util import PiecewiseLinearSchedule

from railrl.torch.vae.conv_vae import ConvVAE, ConvVAETrainer
from railrl.torch.vae.relabeled_vae_experiment import (
    experiment as grill_her_td3_experiment
)
from railrl.torch.vae.tdm_td3_vae_experiment import tdm_td3_vae_experiment


def grill_tdm_td3_full_experiment(variant):
    train_vae_variant = variant['train_vae_variant']
    grill_variant = variant['grill_variant']
    if 'vae_path' not in grill_variant:
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'vae_progress.csv', relative_to_snapshot_dir=True
        )
        vae = train_vae(train_vae_variant)
        rdim = train_vae_variant['representation_size']
        vae_file = logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
        grill_variant['vae_paths'] = {
            str(rdim): vae_file,
        }
        grill_variant['rdim'] = str(rdim)

    logger.remove_tabular_output(
        'vae_progress.csv',
        relative_to_snapshot_dir=True,
    )
    logger.add_tabular_output(
        'progress.csv',
        relative_to_snapshot_dir=True,
    )
    tdm_td3_vae_experiment(variant['grill_variant'])


def grill_her_td3_full_experiment(variant):
    train_vae_variant = variant['train_vae_variant']
    grill_variant = variant['grill_variant']
    if 'vae_path' not in grill_variant:
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'vae_progress.csv', relative_to_snapshot_dir=True
        )
        vae = train_vae(train_vae_variant)
        rdim = train_vae_variant['representation_size']
        vae_file = logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
        grill_variant['vae_paths'] = {
            str(rdim): vae_file,
        }
        grill_variant['rdim'] = str(rdim)

    logger.remove_tabular_output(
        'vae_progress.csv',
        relative_to_snapshot_dir=True,
    )
    logger.add_tabular_output(
        'progress.csv',
        relative_to_snapshot_dir=True,
    )
    grill_her_td3_experiment(variant['grill_variant'])


def train_vae(variant):
    from railrl.core import logger
    import railrl.torch.pytorch_util as ptu
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data, info = variant['generate_vae_fctn'](
        **variant['get_data_kwargs']
    )
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(**variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    m = ConvVAE(representation_size, input_channels=3)
    if ptu.gpu_enabled():
        m.cuda()
    t = ConvVAETrainer(train_data, test_data, m, beta=beta,
                       beta_schedule=beta_schedule, **variant['algo_kwargs'])
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(
            epoch,
            save_reconstruction=should_save_imgs,
            save_scatterplot=should_save_imgs,
            save_vae=False,
        )
        if should_save_imgs:
            t.dump_samples(epoch)
    return m
