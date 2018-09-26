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
from torch.optim import Adam, RMSprop
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
from railrl.torch.vae.skew.common import Dynamics, plot_curves
from railrl.torch.vae.skew.datasets import project_samples_square_np
from railrl.torch.vae.skew.histogram import Histogram, visualize_histogram, \
    visualize_histogram_samples
from railrl.misc.visualization_util import gif

K = 6

"""
Plotting
"""

def visualize_samples(
        epoch, vis_samples_np, encoder, decoder,
        report, dynamics,
        n_vis=1000, xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5)
):
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
    projected_generated_samples = dynamics(generated_samples)
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

    return sample_img


def train_from_variant(variant):
    train(full_variant=variant, **variant)


def train(
        dataset_generator,
        n_start_samples,
        projection=project_samples_square_np,
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
        train_vae_from_histogram=False,
        weight_type='sqrt_inv_p',
        **kwargs
):

    """
    Sanitize Inputs
    """
    encoder = sv.Encoder(
        nn.Linear(2, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, z_dim * 2),
    )
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
    if train_vae_from_histogram:
        assert not weight_loss
        assert not skew_sampling

    report = HTMLReport(
        logger.get_snapshot_dir() + '/report.html',
        images_per_row=5,
    )
    dynamics = Dynamics(projection, dynamics_noise)
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
    heatmap_imgs = []
    sample_imgs = []
    entropies = []
    tvs_to_uniform = []
    entropy_gains_from_reweighting = []
    """
    p_theta = VAE's distribution
    """
    p_theta = Histogram(num_bins, weight_type=weight_type)
    for epoch in sv.progressbar(range(n_epochs)):
        epoch_stats = defaultdict(list)
        p_theta = Histogram(num_bins, weight_type=weight_type)
        vae_samples = sv.generate_vae_samples_np(
            decoder,
            n_samples_to_add_per_epoch,
        )
        p_theta.compute_pvals_and_per_bin_weights(vae_samples)
        projected_samples = dynamics(vae_samples)
        if append_all_data:
            train_data = np.vstack((train_data, projected_samples))
        else:
            train_data = np.vstack((orig_train_data, projected_samples))

        all_weights = p_theta.compute_per_elem_weights(train_data)
        p_new = Histogram(num_bins, weight_type=weight_type)
        p_new.compute_pvals_and_per_bin_weights(
            train_data,
            weights=all_weights,
        )
        all_weights_pt = ptu.np_to_var(all_weights, requires_grad=False)
        if epoch == 0 or (epoch + 1) % save_period == 0:
            epochs.append(epoch)
            encoders.append(copy.deepcopy(encoder))
            decoders.append(copy.deepcopy(decoder))
            report.add_text("Epoch {}".format(epoch))
            heatmap_img = visualize_histogram(epoch, p_theta, report)
            sample_img = visualize_samples(
                epoch, train_data, encoder, decoder, report, dynamics,
            )
            _ = visualize_histogram_samples(
                p_theta, report, dynamics,
                title="P Theta Samples"
            )
            _ = visualize_histogram_samples(
                p_new, report, dynamics,
                title="P New Samples"
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

        """
        train VAE to look like p_new
        """
        if sum(all_weights) == 0:
            all_weights[:] = 1

        indexed_train_data = sv.IndexedData(train_data)
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
        for _, indexed_batch in enumerate(train_dataloader):
            idxs, batch = indexed_batch
            if train_vae_from_histogram:
                batch = ptu.np_to_var(p_new.sample(
                    batch[0].shape[0]
                ))
            else:
                batch = Variable(batch[0].float())

            latents, means, log_vars, stds = encoder.get_encoding_and_suff_stats(
                batch
            )
            beta = float(beta_schedule.get_value(epoch))
            kl = sv.kl_to_prior(means, log_vars, stds)
            reconstruction_log_prob = sv.compute_log_prob(batch, decoder, latents)

            elbo = - kl * beta + reconstruction_log_prob
            if weight_loss:
                # idxs = torch.cat(idxs)
                # batch_weights = all_weights_pt[idxs].unsqueeze(1)
                weights_np = p_theta.compute_per_elem_weights(
                    train_data
                )
                batch_weights = ptu.np_to_var(weights_np)
                loss = -(batch_weights * elbo).mean()
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
        entropies.append(p_theta.entropy())
        tvs_to_uniform.append(p_theta.tv_to_uniform())
        entropy_gain = p_new.entropy() - p_theta.entropy()
        entropy_gains_from_reweighting.append(entropy_gain)

        logger.record_tabular("Epoch", epoch)
        logger.record_tabular("VAE Loss", np.mean(epoch_stats['losses']))
        logger.record_tabular("VAE KL", np.mean(epoch_stats['kls']))
        logger.record_tabular("VAE Log Prob", np.mean(epoch_stats['log_probs']))
        logger.record_tabular('Entropy ', p_theta.entropy())
        logger.record_tabular('KL from uniform', p_theta.kl_from_uniform())
        logger.record_tabular('TV to uniform', p_theta.tv_to_uniform())
        logger.record_tabular('Entropy gain from reweight', entropy_gain)
        logger.dump_tabular()
        logger.save_itr_params(epoch, {
            'encoder': encoder,
            'decoder': decoder,
        })

    report.add_header("Training Curves")
    # results = epochs, encoders, decoders, train_datas, losses, kls, log_probs
    # sv.plot_uniformness(results, full_variant, projection, report=report)
    plot_curves(
        [
            ("Training Loss", losses),
            ("KL", kls),
            ("Log Probs", log_probs),
            ("Entropy Gain from Reweighting", entropy_gains_from_reweighting),
        ],
        report,
    )
    plot_non_vae_curves(entropies, tvs_to_uniform, report)
    report.add_text("Max entropy: {}".format(p_theta.max_entropy()))
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
    gif(
        logger.get_snapshot_dir() + '/samples.gif',
        sample_video,
    )
    gif(
        logger.get_snapshot_dir() + '/heatmaps.gif',
        heatmap_video,
    )
    report.add_image(
        logger.get_snapshot_dir() + '/samples.gif',
        "Samples GIF",
        is_url=True,
    )
    report.add_image(
        logger.get_snapshot_dir() + '/heatmaps.gif',
        "Heatmaps GIF",
        is_url=True,
    )
    report.save()


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
