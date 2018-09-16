"""
Skew the dataset so that it turns into generating a uniform distribution.
"""
import copy
import json
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import chisquare
from torch import nn as nn
from torch.autograd import Variable
from torch.distributions import Normal
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import (
    BatchSampler, WeightedRandomSampler,
    RandomSampler,
)

import railrl.pythonplusplus as ppp
from railrl.core import logger
from railrl.misc import visualization_util as vu
from railrl.misc.html_report import HTMLReport
from railrl.misc.ml_util import ConstantSchedule

"""
Datasets
"""


def gaussian_data(batch_size):
    return np.random.randn(batch_size, 2)


def uniform_truncated_data(batch_size):
    data = np.random.uniform(low=-2, high=2, size=(batch_size, 2))
    data = np.maximum(data, -1)
    data = np.minimum(data, 1)
    return data


def four_corners(_):
    return np.array([
        [-1, 1],
        [-1, -1],
        [1, 1],
        [1, -1],
    ])


def uniform_gaussian_data(batch_size):
    data = np.random.randn(batch_size, 2)
    data = np.maximum(data, -1)
    data = np.minimum(data, 1)
    return data


def uniform_data(batch_size):
    return np.random.uniform(low=-2, high=2, size=(batch_size, 2))


def affine_gaussian_data(batch_size):
    return (
            np.random.randn(batch_size, 2) * np.array([1, 10]) + np.array(
        [20, 1])
    )


def flower_data(batch_size):
    z_true = np.random.uniform(0, 1, batch_size)
    r = np.power(z_true, 0.5)
    phi = 0.25 * np.pi * z_true
    x1 = r * np.cos(phi)
    x2 = r * np.sin(phi)

    # Sampling form a Gaussian
    x1 = np.random.normal(x1, 0.10 * np.power(z_true, 2), batch_size)
    x2 = np.random.normal(x2, 0.10 * np.power(z_true, 2), batch_size)

    # Bringing data in the right form
    X = np.transpose(np.reshape((x1, x2), (2, batch_size)))
    X = np.asarray(X, dtype='float32')
    return X


ut_dataset = uniform_truncated_data(1000)
u_dataset = uniform_data(100)
empty_dataset = np.zeros((0, 2))

"""
Plotting
"""


def show_heatmap(
        encoder, decoder, skew_config,
        xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
        resolution=20,
):
    # encoder, decoder, losses, kls, log_probs = train_results

    def get_prob_batch(batch):
        return compute_train_weights(batch, encoder, decoder, skew_config)

    heat_map = vu.make_heat_map(get_prob_batch, xlim, ylim,
                                resolution=resolution, batch=True)
    # plt.figure()
    vu.plot_heatmap(heat_map)
    # if report:
    #     fig = plt.gcf()
    #     img = vu.save_image(fig)
    #     report.add_image(img, "Weight Heatmap")
    # else:
    #     plt.show()


def plot_weighted_histogram_sample(data, encoder, decoder, skew_config):
    weights_np = compute_train_weights(data, encoder, decoder, skew_config)
    weights = torch.FloatTensor(weights_np)
    samples = torch.multinomial(
        weights, len(weights), replacement=True
    )
    plt.hist(samples, bins=np.arange(0, len(weights)))


def visualize_results(
        results,
        skew_config,
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
        n_vis=1000,
        report=None,
):
    for epoch, encoder, decoder, vis_samples_np in zip(*results[:4]):
        plt.figure()
        show_heatmap(encoder, decoder, skew_config, xlim=xlim, ylim=ylim)
        fig = plt.gcf()
        img = vu.save_image(fig)
        report.add_image(img, "Epoch {} Heatmap".format(epoch))

        plt.figure()
        plot_weighted_histogram_sample(
            vis_samples_np, encoder, decoder, skew_config
        )
        fig = plt.gcf()
        img = vu.save_image(fig)
        report.add_image(img, "Epoch {} Sample Histogram".format(epoch))

        plt.figure()
        plt.suptitle("Epoch {}".format(epoch))

        n_samples = len(vis_samples_np)
        skip_factor = max(n_samples // n_vis, 1)
        vis_samples_np = vis_samples_np[::skip_factor]

        vis_samples = np_to_var(vis_samples_np)
        latents = encoder.encode(vis_samples)
        z_dim = latents.shape[1]
        reconstructed_samples = decoder(latents).data.numpy()
        generated_samples = decoder(
            Variable(torch.randn(n_vis, z_dim))
        ).data.numpy()
        projected_generated_samples = project_samples_np(generated_samples)

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

        if report:
            fig = plt.gcf()
            img = vu.save_image(fig)
            report.add_image(img, "Epoch {}".format(epoch))
    if not report:
        plt.show()


def plot_uniformness(results, n_samples=10000, n_bins=5, report=None):
    generated_frac_on_border_lst = []
    dataset_frac_on_border_lst = []
    p_values = []  # computed using chi squared test
    for epoch, encoder, decoder, vis_samples_np in zip(*results[:4]):

        z_dim = decoder._modules['0'].weight.shape[1]
        generated_samples = decoder(
            Variable(torch.randn(n_samples, z_dim))
        ).data.numpy()
        projected_generated_samples = project_samples_np(generated_samples)

        orig_n_samples_on_border = np.mean(
            np.any(vis_samples_np == 1, axis=1)
            + np.any(vis_samples_np == -1, axis=1)
        )
        dataset_frac_on_border_lst.append(orig_n_samples_on_border)
        gen_n_samples_on_border = np.mean(
            np.any(projected_generated_samples == 1, axis=1)
            + np.any(projected_generated_samples == -1, axis=1)
        )
        generated_frac_on_border_lst.append(gen_n_samples_on_border)

        # Is this data sampled from a uniform distribution? Compute p-value
        h, xe, ye = np.histogram2d(
            projected_generated_samples[:, 0],
            projected_generated_samples[:, 1],
            bins=n_bins,
            range=np.array([[-1, 1], [-1, 1]]),
        )
        counts = h.flatten()
        p_values.append(chisquare(counts).pvalue)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(np.array(generated_frac_on_border_lst))
    plt.xlabel('epoch')
    plt.ylabel('Fraction of points along border')
    plt.title("VAE")

    plt.subplot(1, 3, 2)
    plt.plot(np.array(dataset_frac_on_border_lst))
    plt.xlabel('epoch')
    plt.ylabel('Fraction of points along border')
    plt.title("Training Samples")

    plt.subplot(1, 3, 3)
    plt.plot(np.array(p_values))
    plt.xlabel('epoch')
    plt.ylabel('Uniform Distribution Goodness of fit: p-value')
    plt.title("VAE Samples")
    if report:
        fig = plt.gcf()
        img = vu.save_image(fig)
        report.add_image(img, "uniform-ness")
    else:
        plt.show()


def plot_curves(train_results, report=None):
    *_, losses, kls, log_probs = train_results
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(np.array(losses))
    plt.title("Training Loss")
    plt.subplot(1, 3, 2)
    plt.plot(np.array(kls))
    plt.title("KLs")
    plt.subplot(1, 3, 3)
    plt.plot(np.array(log_probs))
    plt.title("Log Probs")
    plt.xlabel("Epoch")
    if report:
        fig = plt.gcf()
        img = vu.save_image(fig)
        report.add_image(img, "Training curves")
    else:
        plt.show()


def progressbar(it, prefix="", size=60):
    count = len(it)

    def _show(_i):
        x = int(size * _i / count)
        sys.stdout.write(
            "%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), _i, count))
        sys.stdout.flush()

    _show(0)
    for i, item in enumerate(it):
        yield item
        _show(i + 1)
    sys.stdout.write("\n")
    sys.stdout.flush()


"""
VAE specific stuff
"""


def np_to_var(np_array, **kwargs):
    return Variable(torch.from_numpy(np_array).float(), **kwargs)


def kl_to_prior(means, log_stds, stds):
    """
    KL between a Gaussian and a standard Gaussian.

    https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    """
    return 0.5 * (
            - log_stds
            - 1
            + stds ** 2
            + means ** 2
    ).sum(dim=1, keepdim=True)


def log_prob(batch, decoder, latents):
    reconstruction = decoder(latents)
    return -((batch - reconstruction) ** 2).sum(dim=1, keepdim=True)


class Encoder(nn.Sequential):
    def encode(self, x):
        return self.get_encoding_and_suff_stats(x)[0]

    def get_encoding_and_suff_stats(self, x):
        output = self(x)
        z_dim = output.shape[1] // 2
        means, log_stds = (
            output[:, :z_dim], output[:, z_dim:]
        )
        stds = log_stds.exp()
        epsilon = Variable(torch.randn(*means.size()))
        latents = epsilon * stds + means
        return latents, means, log_stds, stds


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.zdim = decoder._modules['0'].weight.shape[1]

    def sample(self, n_samples):
        return self.decoder(
            Variable(torch.randn(n_samples, self.zdim))
        ).data.numpy()


class IndexedData(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


def compute_train_weights(data, encoder, decoder, config):
    """
    :param data: PyTorch
    :param encoder:
    :param decoder:
    :return: PyTorch
    """
    alpha = config.get('alpha', 0)
    mode = config.get('mode', 'none')
    n_average = config.get('n_average', 3)
    temperature = config.get('temperature', 1)
    orig_data_length = len(data)
    if alpha == 0 or mode == 'none':
        return np.ones(orig_data_length)
    data = np.vstack([
        data for _ in range(n_average)
    ])
    data = np_to_var(data)
    z_dim = decoder._modules['0'].weight.shape[1]
    """
    Actually compute the weights
    """
    if mode == 'recon_mse':
        latents, *_ = encoder.get_encoding_and_suff_stats(data)
        reconstruction = decoder(latents)
        weights = ((data - reconstruction) ** 2).sum(dim=1)
    elif mode == 'exp_recon_mse':
        latents, *_ = encoder.get_encoding_and_suff_stats(data)
        reconstruction = decoder(latents)
        weights = (temperature * (data - reconstruction) ** 2).sum(dim=1).exp()
    else:
        if mode == 'biased_encoder':
            latents, means, log_stds, stds = encoder.get_encoding_and_suff_stats(
                data
            )
            importance_weights = 1
        elif mode == 'prior':
            latents = Variable(torch.randn(len(data), z_dim))
            importance_weights = 1
        elif mode == 'importance_sampling':
            latents, means, log_stds, stds = encoder.get_encoding_and_suff_stats(
                data
            )

            prior = Normal(0, 1)
            prior_prob = prior.log_prob(latents).sum(dim=1).exp()

            encoder_distrib = Normal(means, stds)
            encoder_prob = encoder_distrib.log_prob(latents).sum(dim=1).exp()

            importance_weights = prior_prob / encoder_prob
        else:
            raise NotImplementedError()

        data_prob = log_prob(data, decoder, latents).squeeze(1).exp()
        weights = importance_weights * 1. / data_prob
    weights = weights ** alpha

    """
    Average over `n_average`
    """

    samples_of_results = torch.split(weights, orig_data_length, dim=0)
    # pre_avg.shape = ORIG_LEN x N_AVERAGE
    pre_avg = torch.cat(
        [x.unsqueeze(1) for x in samples_of_results],
        1,
    )
    # final.shape = ORIG_LEN
    final = torch.mean(pre_avg, dim=1, keepdim=False)
    return final.data.numpy()


def generate_vae_samples_np(decoder, n_samples):
    z_dim = decoder._modules['0'].weight.shape[1]
    generated_samples = decoder(
        Variable(torch.randn(n_samples, z_dim))
    )
    return generated_samples.data.numpy()


def project_samples_np(samples):
    samples = np.maximum(samples, -1)
    samples = np.minimum(samples, 1)
    return samples


def train_from_variant(variant):
    train(full_variant=variant, **variant)


def train(
        dataset_generator,
        n_start_samples,
        encoder=None,
        decoder=None,
        bs=32,
        n_samples_to_add_per_epoch=1000,
        n_epochs=100,
        skew_config=None,
        weight_loss=False,
        skew_sampling=False,
        beta_schedule=None,
        z_dim=1,
        hidden_size=32,
        save_period=10,
        append_all_data=True,
        full_variant=None,
        **kwargs
):
    if encoder is None:
        encoder = Encoder(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, z_dim * 2),
        )
    if decoder is None:
        decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
    if beta_schedule is None:
        beta_schedule = ConstantSchedule(1)
    if skew_config is None:
        skew_config = dict(
            use_log_prob=False,
            alpha=0,
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
    for epoch in progressbar(range(n_epochs)):
        epoch_stats = defaultdict(list)
        if n_samples_to_add_per_epoch > 0:
            vae_samples = generate_vae_samples_np(decoder,
                                                  n_samples_to_add_per_epoch)
            projected_samples = project_samples_np(vae_samples)
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
        for i, indexed_batch in enumerate(train_dataloader):
            idxs, batch = indexed_batch
            # idxs_torch = torch.cat(idxs)
            # if (epoch+1) % save_period == 0:
            #     # print("i: ", i)
            #     print(
            #         sum(idxs_torch == 0) +
            #         sum(idxs_torch == 1) +
            #         sum(idxs_torch == 2) +
            #         sum(idxs_torch == 3) , " / ", len(batch)
            #     )

            batch = Variable(batch[0].float())

            latents, means, log_stds, stds = encoder.get_encoding_and_suff_stats(
                batch
            )
            beta = float(beta_schedule.get_value(epoch))
            kl = kl_to_prior(means, log_stds, stds)
            reconstruction_log_prob = log_prob(batch, decoder, latents)

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

    report = HTMLReport(
        logger.get_snapshot_dir() + '/report.html',
        images_per_row=3,
    )
    if full_variant:
        report.add_header("Variant")
        report.add_text(
            json.dumps(ppp.dict_to_safe_json(full_variant), indent=2)
        )
    report.add_header("Uniformness")
    plot_uniformness(results, report=report)
    report.add_header("Training Curves")
    plot_curves(results, report=report)
    report.add_header("Samples")
    visualize_results(results, skew_config, report=report)
    report.save()


if __name__ == '__main__':
    plt.close('all')

    # uniform_results = train(
    #     u_dataset,
    #     bs=32,
    #     n_epochs=1000,
    #     n_samples_to_add_per_epoch=0,
    #     skew_config=dict(
    #         alpha=0,
    #         mode='none',
    #         n_average=2,
    #     ),
    #     skew_sampling=False,
    #     z_dim=16,
    #     # beta_schedule=ConstantSchedule(0),
    #     hidden_size=32,
    #     save_period=50,
    # )

    # report = HTMLReport('report_uniform.html')
    # plot_uniformness(uniform_results, report=report)
    # plot_curves(uniform_results, report=report)
    # visualize_results(uniform_results, xlim=(-3, 3), ylim=(-3, 3), report=report)
    # report.save()

    # Skew online dataset to make uniform distribution

    all_results = {}
    for mode in [
        # 'importance_sampling',
        # 'biased_encoder',
        'prior'
    ]:
        all_results[mode] = train(
            ut_dataset[:1000],
            bs=32,
            n_epochs=10,
            n_samples_to_add_per_epoch=10,
            skew_config=dict(
                alpha=1,
                mode=mode,
                n_average=100,
            ),
            skew_sampling=True,
            z_dim=16,
            hidden_size=32,
            save_period=200,
        )

    # report = HTMLReport('report_is.html')
    # plot_uniformness(all_results['importance_sampling'], n_samples=10000,
    #                  n_bins=5, report=report)
    # plot_curves(all_results['importance_sampling'], report=report)
    # visualize_results(all_results['importance_sampling'], report=report)
    # report.save()

    # report = HTMLReport('report_biased.html')
    # plot_uniformness(all_results['biased_encoder'], n_samples=10000,
    #                  n_bins=5, report=report)
    # plot_curves(all_results['biased_encoder'], report=report)
    # visualize_results(all_results['biased_encoder'], report=report)
    # report.save()

    # report = HTMLReport('report_prior_100.html')
    # plot_uniformness(all_results['prior'], n_samples=10000, n_bins=5,
    #                  report=report)
    # plot_curves(all_results['prior'], report=report)
    # visualize_results(all_results['prior'], report=report)
    # report.save()

    # joblib.dump(all_results, "all_results.jb")
    # joblib.dump(all_results, "all_results_prior.jb")
