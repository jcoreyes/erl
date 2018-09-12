"""
Skew the dataset so that it turns into generating a uniform distribution.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.distributions import Normal
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler


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


def compute_train_weights(data, encoder, decoder):
    """
    :param data: PyTorch
    :param encoder:
    :param decoder:
    :return: PyTorch
    """
    data = np_to_var(data)
    latents, means, log_stds, stds = encoder.get_encoding_and_suff_stats(
        data
    )
    decoder_output = decoder(latents)
    decoder_means = decoder_output[:, 0:2]
    decoder_log_stds = decoder_output[:, 2:4]
    distribution = Normal(decoder_means, decoder_log_stds.exp())
    log_prob = distribution.log_prob(data).sum(dim=1)
    prob = log_prob.exp().data.numpy()

    return 1. / np.maximum(prob, 1e-5)


def reconstruction(encoder, decoder, batch):
    latents = encoder.encode(batch)
    return decoder.decode(latents)


def train(data_gen, bs, n_data, n_vis, n_epochs):
    train_data = data_gen(n_data)

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
    for epoch in range(n_epochs):
        print("epoch", epoch)
        weights = compute_train_weights(train_data, encoder, decoder)
        if sum(weights) == 0:
            weights[:] = 1
        print("----")
        print("weight mean", weights.mean())
        print("weight max", weights.max())
        print("weight min", weights.min())
        print("weight std", weights.std())
        base_sampler = WeightedRandomSampler(weights, len(weights))
        train_dataloader = DataLoader(
            train_data,
            sampler=BatchSampler(
                base_sampler,
                batch_size=bs,
                drop_last=False,
            ),
        )
        for i, batch in enumerate(train_dataloader):
            batch = Variable(batch[0].float())

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
    vis_samples_np = data_gen(n_vis)
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
    bs = 32
    n_data = 10000
    n_vis = 1000
    n_epochs = 10
    train(gaussian_data, bs, n_data, n_vis, n_epochs)
    # train(flower_data)
