"""
VAE on the swirl task
"""
from torch.autograd import Variable
from torch.distributions import Normal
from torch.optim import Adam
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn as nn
import torch.nn.functional as F


SWIRL_RATE = 1
T = 10
def swirl_data(batch_size):
    t = np.random.uniform(size=batch_size, low=0, high=T)
    x = t * np.cos(t * SWIRL_RATE) / T
    y = t * np.sin(t * SWIRL_RATE) / T
    data = np.array([x, y]).T
    noise = np.random.randn(batch_size, 2) / (T * 2)
    return data + noise, t


def swirl_t_to_data(t):
    x = t * np.cos(t * SWIRL_RATE) / T
    y = t * np.sin(t * SWIRL_RATE) / T
    return np.array([x, y]).T


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
N_BATCHES = 10000
N_VIS = 1000
HIDDEN_SIZE = 32


def train(data_gen):
    encoder = Encoder(
        nn.Linear(1, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, 2),
    )
    decoder = Decoder(
        nn.Linear(1, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, 4),
    )
    # decoder = Decoder(
    #     nn.Linear(1, 10),
    #     nn.ReLU(),
    #     nn.Linear(10, 10),
    #     nn.ReLU(),
    #     nn.Linear(10, 4),
    # )
    encoder_opt = Adam(encoder.parameters())
    decoder_opt = Adam(decoder.parameters())

    losses = []
    kls = []
    log_probs = []
    for _ in range(N_BATCHES):
        # batch = data_gen(BS)
        batch, true_latents = data_gen(BS)
        batch = np_to_var(batch)
        true_latents = np_to_var(true_latents)

        latents, means, log_stds, stds = encoder.get_encoding_and_suff_stats(
            batch
        )
        kl = kl_to_prior(means, log_stds, stds)

        latent_loss = ((true_latents - latents)**2).mean()

        decoder_output = decoder(latents.detach())
        # decoder_output = decoder(true_latents.unsqueeze(1))
        decoder_means = decoder_output[:, 0:2]
        decoder_log_stds = decoder_output[:, 2:4]
        distribution = Normal(decoder_means, decoder_log_stds.exp())
        reconstruction_log_prob = distribution.log_prob(batch).sum(dim=1)

        # loss = - reconstruction_log_prob.mean()
        # decoder_opt.zero_grad()
        # loss.backward()
        # decoder_opt.step()
        #
        # losses.append(loss.data.numpy())
        # kls.append(0)
        # log_probs.append(reconstruction_log_prob.mean().data.numpy())

        # elbo = - kl + reconstruction_log_prob
        # loss = - elbo.mean()
        loss = - reconstruction_log_prob.mean() + latent_loss
        decoder_opt.zero_grad()
        encoder_opt.zero_grad()
        loss.backward()
        decoder_opt.step()
        encoder_opt.step()

        losses.append(loss.data.numpy())
        kls.append(kl.mean().data.numpy())
        log_probs.append(reconstruction_log_prob.mean().data.numpy())

    # Visualize
    vis_samples_np, true_latents_np = data_gen(N_VIS)
    vis_samples = np_to_var(vis_samples_np)
    true_latents = np_to_var(true_latents_np).unsqueeze(1)
    true_means = swirl_t_to_data(true_latents_np)
    import ipdb; ipdb.set_trace()
    latents = encoder.encode(vis_samples)
    reconstructed_samples = decoder.decode(latents).data.numpy()
    # reconstructed_samples = decoder.decode(true_latents).data.numpy()
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
    estimated_means = swirl_t_to_data(latents.data.numpy())
    plt.plot(estimated_means[:, 0], estimated_means[:, 1], '.')
    plt.legend(["Samples", "Projected Latents"])
    plt.title("Reconstruction")
    plt.subplot(2, 3, 6)
    plt.plot(vis_samples_np[:, 0], vis_samples_np[:, 1], '.')
    plt.plot(true_means[:, 0], true_means[:, 1], '.')
    plt.title("Original Samples")
    plt.legend(["Original", "True means"])
    plt.show()


if __name__ == '__main__':
    # train(gaussian_data)
    # train(flower_data)
    train(swirl_data)
