import copy
import json
from collections import defaultdict

import numpy as np
import torch
from skvideo.io import vwrite
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import (
    BatchSampler, WeightedRandomSampler,
    RandomSampler,
)

import railrl.pythonplusplus as ppp
from railrl.core import logger
from railrl.misc.html_report import HTMLReport


class Histogram(object):
    """
    A perfect histogram
    """

    def __init__(self, num_bins):
        self.pvals = np.zeros(num_bins)

    def sample(self, n_samples):
        return np.random.multinomial(n_samples, self.pvals)


def train(
        dataset_generator,
        n_start_samples,
        projection=project_samples_square_np,
        histogram=None,
        bs=32,
        n_samples_to_add_per_epoch=1000,
        n_epochs=100,
        skew_config=None,
        weight_loss=False,
        skew_sampling=False,
        save_period=10,
        append_all_data=True,
        full_variant=None,
        dynamics_noise=0,
        **kwargs
):
    if histogram is None:
        encoder = Histogram()
    if skew_config is None:
        skew_config = dict(
            use_log_prob=False,
            alpha=0,
        )
    report = HTMLReport(
        logger.get_snapshot_dir() + '/report.html',
        images_per_row=3,
        )
    if full_variant:
        report.add_header("Variant")
        report.add_text(
            json.dumps(
                ppp.dict_to_safe_json(
                    full_variant,
                    sort=True),
                indent=2,
            )
        )

    orig_train_data = dataset_generator(n_start_samples)
    train_data = orig_train_data

    histograms = []
    train_datas = []
    heatmap_imgs = []
    sample_imgs = []
    for epoch in range(n_epochs):
        epoch_stats = defaultdict(list)
        if n_samples_to_add_per_epoch > 0:
            samples = histogram.sample(n_samples_to_add_per_epoch)

            new_samples = samples + dynamics_noise * np.random.randn(
                *samples.shape
            )
            projected_samples = projection(new_samples)
            if append_all_data:
                train_data = np.vstack((train_data, projected_samples))
            else:
                train_data = np.vstack((orig_train_data, projected_samples))
        indexed_train_data = IndexedData(train_data)

        all_weights = compute_train_weights(train_data, encoder, decoder,
                                            skew_config)
        all_weights_pt = np_to_var(all_weights, requires_grad=False)
        if sum(all_weights) == 0:
            all_weights[:] = 1

        if skew_sampling:
            base_sampler = WeightedRandomSampler(all_weights, len(all_weights))
        else:
            base_sampler = RandomSampler(indexed_train_data)

        train_dataloader = DataLoader(
            indexed_train_data,
            sampler=BatchSampler(
                base_sampler,
                batch_size=bs,
                drop_last=False,
            ),
        )
        if epoch == 0 or (epoch + 1) % save_period == 0:
            epochs.append(epoch)
            encoders.append(copy.deepcopy(encoder))
            decoders.append(copy.deepcopy(decoder))
            train_datas.append(train_data)
            heatmap_img, sample_img = (
                visualize(epoch, train_data, encoder, decoder, full_variant,
                          report, projection)
            )
            heatmap_imgs.append(heatmap_img)
            sample_imgs.append(sample_img)
            report.save()

            from PIL import Image
            Image.fromarray(heatmap_img).save(
                logger.get_snapshot_dir() + '/heatmap{}.png'.format(epoch)
            )
            Image.fromarray(sample_img).save(
                logger.get_snapshot_dir() + '/samples{}.png'.format(epoch)
            )
        for i, indexed_batch in enumerate(train_dataloader):
            idxs, batch = indexed_batch
            batch = Variable(batch[0].float())

            latents, means, log_vars, stds = encoder.get_encoding_and_suff_stats(
                batch
            )
            beta = float(beta_schedule.get_value(epoch))
            kl = kl_to_prior(means, log_vars, stds)
            reconstruction_log_prob = compute_log_prob(batch, decoder, latents)

            elbo = - kl * beta + reconstruction_log_prob
            if weight_loss:
                idxs = torch.cat(idxs)
                weights = all_weights_pt[idxs].unsqueeze(1)
                loss = -(weights * elbo).mean()
            else:
                loss = - elbo.mean()
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            loss.backward()
            encoder_opt.step()
            decoder_opt.step()

            epoch_stats['losses'].append(loss.data.numpy())
            epoch_stats['kls'].append(kl.mean().data.numpy())
            epoch_stats['log_probs'].append(
                reconstruction_log_prob.mean().data.numpy()
            )

        losses.append(np.mean(epoch_stats['losses']))
        kls.append(np.mean(epoch_stats['kls']))
        log_probs.append(np.mean(epoch_stats['log_probs']))

        logger.record_tabular("Epoch", epoch)
        logger.record_tabular("Loss", np.mean(epoch_stats['losses']))
        logger.record_tabular("KL", np.mean(epoch_stats['kls']))
        logger.record_tabular("Log Prob", np.mean(epoch_stats['log_probs']))
        logger.dump_tabular()
        logger.save_itr_params(epoch, {
            'encoder': encoder,
            'decoder': decoder,
        })

    results = epochs, encoders, decoders, train_datas, losses, kls, log_probs
    report.add_header("Uniformness")
    plot_uniformness(results, full_variant, projection, report=report)
    report.add_header("Training Curves")
    plot_curves(results, report=report)
    report.save()

    heatmap_video = np.stack(heatmap_imgs)
    sample_video = np.stack(sample_imgs)

    vwrite(
        logger.get_snapshot_dir() + '/heatmaps.mp4',
        heatmap_video,
        )
    vwrite(
        logger.get_snapshot_dir() + '/samples.mp4',
        sample_video,
        )
