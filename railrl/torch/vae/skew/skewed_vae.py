"""
Skew the dataset so that it turns into generating a uniform distribution.
"""
import copy
import json
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from skvideo.io import vwrite
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
from railrl.torch.core import PyTorchModule
from railrl.torch.vae.skew.datasets import project_samples_square_np
import railrl.torch.pytorch_util as ptu

K = 6

"""
Plotting
"""


def plot_weighted_histogram_sample(data, encoder, decoder, skew_config):
    weights_np = compute_train_weights(data, encoder, decoder, skew_config)
    weights = torch.FloatTensor(weights_np)
    samples = torch.multinomial(
        weights, len(weights), replacement=True
    )
    plt.hist(samples, bins=np.arange(0, len(weights)))


def get_weight_stats(data, encoder, decoder, skew_config):
    weights_np = compute_train_weights(data, encoder, decoder, skew_config)
    stats = []
    stats.append("weight avg: {:4.4f}".format(weights_np.mean()))
    stats.append("weight std: {:4.4f}".format(weights_np.std()))
    stats.append("weight min: {:4.4f}".format(weights_np.min()))
    stats.append("weight max: {:4.4f}".format(weights_np.max()))

    # bottom_k = np.argpartition(weights_np, K)[:K]
    # top_k = np.argpartition(-weights_np, K)[:K]
    arg_sorted = weights_np.argsort()
    top_k = arg_sorted[-K:][::-1]
    bottom_k = arg_sorted[:K]
    for rank, i in enumerate(top_k):
        stats.append('max {}. index = {}. value = {:4.4f}. pos = {}'.format(
            rank,
            i,
            weights_np[i],
            data[i]
        ))
    for rank, i in enumerate(bottom_k):
        stats.append('min {} index. value = {:4.4f}. pos = {}'.format(
            rank,
            weights_np[i],
            data[i]
        ))
    return '\n'.join(stats)


def visualize(epoch, vis_samples_np, encoder, decoder, full_variant,
              report, projection, n_vis=1000, xlim=(-1.5, 1.5),
              ylim=(-1.5, 1.5)):
    report.add_text("Epoch {}".format(epoch))

    skew_config = full_variant['skew_config']
    dynamics_noise = full_variant['dynamics_noise']
    plt.figure()
    show_weight_heatmap(encoder, decoder, skew_config, xlim=xlim, ylim=ylim)
    fig = plt.gcf()
    heatmap_img = vu.save_image(fig)
    report.add_image(heatmap_img, "Epoch {} Heatmap".format(epoch))


    plt.figure()
    plt.suptitle("Epoch {}".format(epoch))
    n_samples = len(vis_samples_np)
    skip_factor = max(n_samples // n_vis, 1)
    vis_samples_np = vis_samples_np[::skip_factor]
    vis_samples = np_to_var(vis_samples_np)
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

    weight_stats = get_weight_stats(
        vis_samples_np, encoder, decoder, skew_config,
    )
    report.add_text(weight_stats)

    return heatmap_img, sample_img


def plot_uniformness(results, full_variant, projection, n_samples=10000,
                     n_bins=5,
                     report=None):
    dynamics_noise = full_variant['dynamics_noise']
    generated_frac_on_border_lst = []
    dataset_frac_on_border_lst = []
    p_values = []  # computed using chi squared test
    for epoch, encoder, decoder, vis_samples_np in zip(*results[:4]):

        z_dim = decoder._modules['0'].weight.shape[1]
        generated_samples = decoder.reconstruct(
            Variable(torch.randn(n_samples, z_dim))
        ).data.numpy()
        generated_samples = generated_samples + dynamics_noise * np.random.randn(
            *generated_samples.shape
        )
        projected_generated_samples = projection(
            generated_samples,
        )

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


def kl_to_prior(means, log_vars, stds):
    """
    KL between a Gaussian and a standard Gaussian.

    https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    """
    # Implement for one dimension. Broadcasting will take care of the constants.
    return 0.5 * (
            - log_vars
            - 1
            + 2 * (stds ** 2)
            + means ** 2
    ).sum(dim=1, keepdim=True)


def compute_log_prob(batch, decoder, latents):
    mu, var = decoder.decode(latents)
    dist = Normal(mu, var.pow(0.5))
    vals = dist.log_prob(batch).sum(dim=1, keepdim=True)
    return vals


def all_finite(x):
    """
    Quick pytorch test that there are no nan's or infs.

    note: torch now has torch.isnan
    url: https://gist.github.com/wassname/df8bc03e60f81ff081e1895aabe1f519
    """
    not_inf = ((x + 1) != x)
    not_nan = (x == x)
    is_finite_vector = not_inf & not_nan
    return is_finite_vector.int().data.numpy().all()


class Encoder(nn.Sequential):
    def encode(self, x):
        return self.get_encoding_and_suff_stats(x)[0]

    def get_encoding_and_suff_stats(self, x):
        output = self(x)
        z_dim = output.shape[1] // 2
        means, log_var = (
            output[:, :z_dim], output[:, z_dim:]
        )
        stds = (0.5 * log_var).exp()
        epsilon = Variable(torch.randn(*means.size()))
        latents = epsilon * stds + means
        return latents, means, log_var, stds


class Decoder(nn.Sequential):
    def __init__(self, *args, output_var=1, output_offset=0):
        super().__init__(*args)
        self.output_var = output_var
        self.output_offset = output_offset

    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        return output + self.output_offset

    def decode(self, input):
        output = self(input)
        if self.output_var == 'learned':
            mu, logvar = torch.split(output, 2, dim=1)
            var = logvar.exp()
        else:
            mu = output
            var = self.output_var * torch.ones_like(mu)
        return mu, var

    def reconstruct(self, input):
        mu, _ = self.decode(input)
        return mu

    def sample(self, latent):
        mu, var = self.decode(latent)
        return Normal(mu, var.pow(0.5)).sample()


class VAE(PyTorchModule):
    def __init__(
            self,
            encoder,
            decoder,
            z_dim,
            mode='importance_sampling',
            min_prob=1e-7,
            n_average=100,
    ):
        self.quick_init(locals())
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.z_dim = z_dim
        self.mode = mode
        self.min_log_prob = np.log(min_prob)
        self.n_average = n_average

    def sample_given_z(self, latent):
        return self.decoder.sample(latent)

    def sample(self, num_samples):
        return self.sample_given_z(
            Variable(torch.randn(num_samples, self.z_dim))
        ).data.numpy()

    def reconstruct(self, data):
        latents = self.encoder.encode(ptu.np_to_var(data))
        return self.decoder.reconstruct(latents).data.numpy()

    def compute_density(self, data):
        orig_data_length = len(data)
        data = np.vstack([
            data for _ in range(self.n_average)
        ])
        data = np_to_var(data)
        if self.mode == 'biased_encoder':
            latents, means, log_vars, stds = (
                self.encoder.get_encoding_and_suff_stats(data)
            )
            importance_weights = 1
        elif self.mode == 'prior':
            latents = Variable(torch.randn(len(data), self.z_dim))
            importance_weights = 1
        elif self.mode == 'importance_sampling':
            latents, means, log_vars, stds = (
                self.encoder.get_encoding_and_suff_stats(data)
            )
            prior = Normal(0, 1)
            prior_log_prob = prior.log_prob(latents).sum(dim=1)

            encoder_distrib = Normal(means, stds)
            encoder_log_prob = encoder_distrib.log_prob(latents).sum(dim=1)

            importance_weights = (prior_log_prob - encoder_log_prob).exp()
        else:
            raise NotImplementedError()

        data_log_prob = compute_log_prob(data, self.decoder, latents).squeeze(1)
        # data_log_prob = torch.clamp(
        #     data_log_prob,
        #     min=self.min_log_prob,
        # )
        unweighted_data_prob = data_log_prob.exp()
        dp = unweighted_data_prob.data.numpy()

        print("---------")
        print("max dp", np.max(dp))
        print("min dp", np.min(dp))
        print("std dp", np.std(dp))
        print("mean dp", np.mean(dp))

        data_prob = importance_weights * unweighted_data_prob / importance_weights.sum()
        data_prob = data_prob * self.n_average

        iw = importance_weights.data.numpy()
        print("----")
        print("max iw", np.max(iw))
        print("min iw", np.min(iw))
        print("std iw", np.std(iw))
        print("mean iw", np.mean(iw))


        """
        Average over `n_average`
        """
        samples_of_results = torch.split(data_prob, orig_data_length, dim=0)
        # pre_avg.shape = ORIG_LEN x N_AVERAGE
        pre_avg = torch.stack(samples_of_results, dim=1)
        # final.shape = ORIG_LEN
        final = torch.mean(pre_avg, dim=1, keepdim=False)
        return final.data.numpy()


class IndexedData(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


def train_from_variant(variant):
    train(full_variant=variant, **variant)


def train(
        dataset_generator,
        n_start_samples,
        projection=project_samples_square_np,
        encoder=None,
        decoder=None,
        bs=32,
        n_samples_to_add_per_epoch=1000,
        n_epochs=100,
        skew_config=None,
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
        if decoder_output_var == 'learned':
            last_layer = nn.Linear(hidden_size, 4)
        else:
            last_layer = nn.Linear(hidden_size, 2)
        decoder = Decoder(
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
    for epoch in progressbar(range(n_epochs)):
        epoch_stats = defaultdict(list)
        if n_samples_to_add_per_epoch > 0:
            vae_samples = generate_vae_samples_np(decoder,
                                                  n_samples_to_add_per_epoch)

            new_samples = vae_samples + dynamics_noise * np.random.randn(
                *vae_samples.shape
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
