"""
Skew the dataset so that it turns into generating a uniform distribution.
"""
import copy
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skvideo.io import vwrite
from torch import nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import (
    BatchSampler, WeightedRandomSampler,
    RandomSampler,
)

import railrl.pythonplusplus as ppp
import railrl.torch.pytorch_util as ptu
import railrl.torch.vae.skew.skewed_vae as sv
from railrl.core import logger
from railrl.misc import visualization_util as vu
from railrl.misc.html_report import HTMLReport
from railrl.misc.ml_util import ConstantSchedule
from railrl.torch.vae.skew.datasets import project_samples_square_np
from railrl.torch.vae.skew.histogram import Histogram

K = 6

"""
Plotting
"""

def visualize(epoch, vis_samples_np, encoder, decoder, full_variant,
              report, projection,
              histogram,
              n_vis=1000, xlim=(-1.5, 1.5),
              ylim=(-1.5, 1.5)):
    dynamics_noise = full_variant['dynamics_noise']
    report.add_text("Epoch {}".format(epoch))

    plt.figure()
    fig = plt.gcf()
    ax = plt.gca()
    heatmap_img = ax.imshow(
        np.swapaxes(histogram.pvals, 0, 1),  # imshow uses first axis as y-axis
        extent=[-1, 1, -1, 1],
        cmap=plt.get_cmap('plasma'),
        interpolation='nearest',
        aspect='auto',
        origin='bottom',  # <-- Important! By default top left is (0, 0)
    )
    divider = make_axes_locatable(ax)
    legend_axis = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(heatmap_img, cax=legend_axis, orientation='vertical')
    heatmap_img = vu.save_image(fig)
    if histogram.num_bins < 5:
        pvals_str = np.array2string(histogram.pvals, precision=3)
        report.add_text(pvals_str)
    report.add_image(heatmap_img, "Epoch {} Heatmap".format(epoch))

    plt.figure()
    fig = plt.gcf()
    ax = plt.gca()
    heatmap_img = ax.imshow(
        np.swapaxes(histogram.weights, 0, 1),  # imshow uses first axis as
        # y-axis
        extent=[-1, 1, -1, 1],
        cmap=plt.get_cmap('plasma'),
        interpolation='nearest',
        aspect='auto',
        origin='bottom',  # <-- Important! By default top left is (0, 0)
    )
    divider = make_axes_locatable(ax)
    legend_axis = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(heatmap_img, cax=legend_axis, orientation='vertical')
    heatmap_img = vu.save_image(fig)
    if histogram.num_bins < 5:
        pvals_str = np.array2string(histogram.pvals, precision=3)
        report.add_text(pvals_str)
    report.add_image(heatmap_img, "Epoch {} Weight Heatmap".format(epoch))


    plt.figure()
    plt.suptitle("Epoch {}".format(epoch))
    n_samples = len(vis_samples_np)
    skip_factor = max(n_samples // n_vis, 1)
    vis_samples_np = vis_samples_np[::skip_factor]
    vis_samples = ptu.np_to_var(vis_samples_np)
    latents = encoder.encode(vis_samples)
    z_dim = latents.shape[1]
    reconstructed_samples = decoder.reconstruct(latents).data.numpy()
    generated_samples = decoder.reconstruct(
        Variable(torch.randn(n_vis, z_dim))
    ).data.numpy()
    generated_samples = generated_samples + dynamics_noise * np.random.randn(
        *generated_samples.shape
    )
    projected_generated_samples = projection(
        generated_samples,
    )
    plt.subplot(2, 2, 1)
    plt.plot(generated_samples[:, 0], generated_samples[:, 1], '.')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title("Generated Samples")
    plt.subplot(2, 2, 2)
    plt.plot(projected_generated_samples[:, 0],
             projected_generated_samples[:, 1], '.')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title("Projected Generated Samples")
    plt.subplot(2, 2, 3)
    plt.plot(reconstructed_samples[:, 0], reconstructed_samples[:, 1], '.')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title("Reconstruction")
    plt.subplot(2, 2, 4)
    plt.plot(vis_samples_np[:, 0], vis_samples_np[:, 1], '.')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title("Original Samples")

    fig = plt.gcf()
    sample_img = vu.save_image(fig)
    report.add_image(sample_img, "Epoch {} Samples".format(epoch))

    return heatmap_img, sample_img


def train_from_variant(variant):
    train(full_variant=variant, **variant)


def train(
        dataset_generator,
        n_start_samples,
        projection=project_samples_square_np,
        encoder=None,
        decoder=None,
        histogram=None,
        bs=32,
        n_samples_to_add_per_epoch=1000,
        n_epochs=100,
        weight_loss=False,
        skew_sampling=False,
        beta_schedule_class=None,
        beta_schedule_kwargs=None,
        z_dim=1,
        hidden_size=32,
        save_period=10,
        append_all_data=True,
        full_variant=None,
        dynamics_noise=0,
        decoder_output_var='learned',
        num_bins=5,
        **kwargs
):
    if histogram is None:
        histogram = Histogram(num_bins)
    if encoder is None:
        encoder = sv.Encoder(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, z_dim * 2),
        )
    if decoder is None:
        if decoder_output_var == 'learned':
            last_layer = nn.Linear(hidden_size, 4)
        else:
            last_layer = nn.Linear(hidden_size, 2)
        decoder = sv.Decoder(
            nn.Linear(z_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            last_layer,
            output_var=decoder_output_var,
        )
    if beta_schedule_class is None:
        beta_schedule = ConstantSchedule(1)
    else:
        beta_schedule = beta_schedule_class(**beta_schedule_kwargs)
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

    encoder_opt = Adam(encoder.parameters())
    decoder_opt = Adam(decoder.parameters())

    epochs = []
    losses = []
    kls = []
    log_probs = []
    encoders = []
    decoders = []
    train_datas = []
    heatmap_imgs = []
    sample_imgs = []
    entropies = []
    tvs_to_uniform = []
    for epoch in sv.progressbar(range(n_epochs)):
        epoch_stats = defaultdict(list)
        if n_samples_to_add_per_epoch > 0:
            vae_samples = sv.generate_vae_samples_np(
                decoder,
                n_samples_to_add_per_epoch,
            )

            new_samples = vae_samples + dynamics_noise * np.random.randn(
                *vae_samples.shape
            )
            projected_samples = projection(new_samples)
            if append_all_data:
                train_data = np.vstack((train_data, projected_samples))
            else:
                train_data = np.vstack((orig_train_data, projected_samples))
        indexed_train_data = sv.IndexedData(train_data)

        histogram.compute_pvals_and_weights(train_data)
        all_weights = histogram.compute_weights(train_data)
        all_weights_pt = ptu.np_to_var(all_weights, requires_grad=False)
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
                             report, projection, histogram)
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
            kl = sv.kl_to_prior(means, log_vars, stds)
            reconstruction_log_prob = sv.compute_log_prob(batch, decoder, latents)

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
        entropies.append(histogram.entropy())
        tvs_to_uniform.append(histogram.tv_to_uniform())

        logger.record_tabular("Epoch", epoch)
        logger.record_tabular("VAE Loss", np.mean(epoch_stats['losses']))
        logger.record_tabular("VAE KL", np.mean(epoch_stats['kls']))
        logger.record_tabular("VAE Log Prob", np.mean(epoch_stats['log_probs']))
        logger.record_tabular('Entropy ', histogram.entropy())
        logger.record_tabular('KL from uniform', histogram.kl_from_uniform())
        logger.record_tabular('Tv to uniform', histogram.tv_to_uniform())
        logger.dump_tabular()
        logger.save_itr_params(epoch, {
            'encoder': encoder,
            'decoder': decoder,
        })

    report.add_header("Training Curves")
    results = epochs, encoders, decoders, train_datas, losses, kls, log_probs
    sv.plot_uniformness(results, full_variant, projection, report=report)
    sv.plot_curves(results, report=report)
    plot_non_vae_curves(entropies, tvs_to_uniform, report)
    report.add_text("Max entropy: {}".format(histogram.max_entropy()))
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


def plot_non_vae_curves(entropies, tvs_to_uniform, report):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(np.array(entropies))
    plt.title("Entropy")
    plt.subplot(1, 2, 2)
    plt.plot(np.array(tvs_to_uniform))
    plt.title("TV to uniform")
    plt.xlabel("Epoch")
    fig = plt.gcf()
    img = vu.save_image(fig)
    report.add_image(img, "Final Distribution")
