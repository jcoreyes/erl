"""
Example VAE on gaussian, flower, and swirl data (artificially generated).

Should take about 20 seconds to run. The Gaussian and Flower datasets work,
but not the swirl one.
"""
from torch.autograd import Variable
from torch.distributions import Normal
from torch.optim import Adam
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn as nn
import torch.nn.functional as F


def gaussian_data(batch_size):
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


def np_to_var(np_array):
    return Variable(torch.from_numpy(np_array).float())


def kl_to_prior(means, log_stds, stds):
    """
    KL between a Gaussian and a standard Gaussian.

    https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    """
    return 0.5 * (
            - 2 * log_stds  # log std_prior = 0
            - 1  # d = 1
            + stds ** 2
            + means ** 2
    )


class Encoder(nn.Sequential):
    def encode(self, x):
        return self.get_encoding_and_suff_stats(x)[0]

    def get_encoding_and_suff_stats(self, x):
        output = self(x)
        means, log_stds = (
            output[:, 0], output[:, 1]
        )
        stds = log_stds.exp()
        epsilon = Variable(torch.randn(*means.size()))
        latents = epsilon * stds + means
        latents = latents.unsqueeze(1)
        return latents, means, log_stds, stds


class Decoder(nn.Sequential):
    def decode(self, latents):
        output = self(latents)
        means, log_stds = output[:, 0:2], output[:, 2:4]
        distribution = Normal(means, log_stds.exp())
        return distribution.sample()


BS = 16
N_BATCHES = 100
N_VIS = 100


def train(data_gen):
    encoder = Encoder(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
    )
    decoder = Decoder(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    )
    encoder_opt = Adam(encoder.parameters())
    decoder_opt = Adam(decoder.parameters())

    losses = []
    kls = []
    log_probs = []
    for _ in range(N_BATCHES):
        batch = data_gen(BS)
        batch = np_to_var(batch)

        latents, means, log_stds, stds = encoder.get_encoding_and_suff_stats(
            batch
        )
        kl = kl_to_prior(means, log_stds, stds)

        decoder_output = decoder(latents)
        decoder_means = decoder_output[:, 0:2]
        decoder_log_stds = decoder_output[:, 2:4]
        distribution = Normal(decoder_means, decoder_log_stds.exp())
        reconstruction_log_prob = distribution.log_prob(batch).sum(dim=1)

        elbo = - kl + reconstruction_log_prob

        loss = - elbo.mean()
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        loss.backward()
        encoder_opt.step()
        decoder_opt.step()

        losses.append(loss.data.numpy())
        kls.append(kl.mean().data.numpy())
        log_probs.append(reconstruction_log_prob.mean().data.numpy())

    # Visualize
    vis_samples_np = data_gen(N_VIS)
    vis_samples = np_to_var(vis_samples_np)
    latents = encoder.encode(vis_samples)
    reconstructed_samples = decoder.decode(latents).data.numpy()
    generated_samples = decoder.decode(
        Variable(torch.randn(*latents.shape))
    ).data.numpy()

    plt.subplot(2, 3, 1)
    plt.plot(np.array(losses))
    plt.title("Training Loss")
    plt.subplot(2, 3, 2)
    plt.plot(np.array(kls))
    plt.title("KLs")
    plt.subplot(2, 3, 3)
    plt.plot(np.array(log_probs))
    plt.title("Log Probs")

    plt.subplot(2, 3, 4)
    plt.plot(generated_samples[:, 0], generated_samples[:, 1], '.')
    plt.title("Generated Samples")
    plt.subplot(2, 3, 5)
    plt.plot(reconstructed_samples[:, 0], reconstructed_samples[:, 1], '.')
    plt.title("Reconstruction")
    plt.subplot(2, 3, 6)
    plt.plot(vis_samples_np[:, 0], vis_samples_np[:, 1], '.')
    plt.title("Original Samples")
    plt.show()


if __name__ == '__main__':
    # train(gaussian_data)
    train(flower_data)
