"""
VAE on the swirl task.

Basically, VAEs don't work. It's probably because the prior isn't very good
and/or because the learning signal is pretty weak when both the encoder and
decoder change quickly. However, I tried also alternating between the two,
and that didn't seem to help.
"""
from torch.distributions import Normal
from torch.optim import Adam
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn as nn
import rlkit.torch.pytorch_util as ptu

SWIRL_RATE = 1
T = 10
BS = 128
N_BATCHES = 2000
N_VIS = 1000
HIDDEN_SIZE = 32
VERBOSE = False


def swirl_data(batch_size):
    t = np.random.uniform(size=batch_size, low=0, high=T)
    x = t * np.cos(t * SWIRL_RATE) / T
    y = t * np.sin(t * SWIRL_RATE) / T
    data = np.array([x, y]).T
    noise = np.random.randn(batch_size, 2) / (T * 2)
    return data + noise, t.reshape(-1, 1)


def swirl_t_to_data(t):
    x = t * np.cos(t * SWIRL_RATE) / T
    y = t * np.sin(t * SWIRL_RATE) / T
    return np.array([x, y]).T


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
            output[:, 0:1], output[:, 1:2]
        )
        stds = log_stds.exp()
        epsilon = ptu.Variable(torch.randn(*means.size()))
        latents = epsilon * stds + means
        latents = latents
        return latents, means, log_stds, stds


class Decoder(nn.Sequential):
    def decode(self, latents):
        output = self(latents)
        means, log_stds = output[:, 0:2], output[:, 2:4]
        distribution = Normal(means, log_stds.exp())
        return distribution.sample()


def t_to_xy(t):
    if len(t.shape) == 2:
        t = t[:, 0]
    x = t * np.cos(t * SWIRL_RATE) / T
    y = t * np.sin(t * SWIRL_RATE) / T
    return np.array([x, y]).T


def pretrain_encoder(encoder, opt):
    losses = []
    for _ in range(1000):
        x_np, y_np = swirl_data(BS)
        x = ptu.np_to_var(x_np)
        y = ptu.np_to_var(y_np)
        y_hat = encoder.encode(x)
        loss = ((y_hat - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.data.numpy())

    if VERBOSE:
        x_np, y_np = swirl_data(N_VIS)
        x = ptu.np_to_var(x_np)
        y_hat = encoder.encode(x)
        y_hat_np = y_hat.data.numpy()
        x_hat_np = t_to_xy(y_hat_np[:, 0])

        plt.subplot(2, 1, 1)
        plt.plot(np.array(losses))
        plt.title("Training Loss")

        plt.subplot(2, 1, 2)
        plt.plot(x_np[:, 0], x_np[:, 1], '.')
        plt.plot(x_hat_np[:, 0], x_hat_np[:, 1], '.')
        plt.title("Samples")
        plt.legend(["Samples", "Estimates"])
        plt.show()


def train_encoder(encoder, decoder, encoder_opt):
    batch, true_latents = swirl_data(BS)
    batch = ptu.np_to_var(batch)

    latents, means, log_stds, stds = encoder.get_encoding_and_suff_stats(
        batch
    )
    kl = kl_to_prior(means, log_stds, stds)

    latents = encoder.encode(batch)
    decoder_output = decoder(latents)
    decoder_means = decoder_output[:, 0:2]
    decoder_log_stds = decoder_output[:, 2:4]
    distribution = Normal(decoder_means, decoder_log_stds.exp())
    reconstruction_log_prob = distribution.log_prob(batch).sum(dim=1)

    # elbo = - kl + reconstruction_log_prob
    # loss = - elbo.mean()
    loss = - reconstruction_log_prob.mean()
    # This is the second place where we cheat:
    latent_loss = ((ptu.np_to_var(true_latents) - latents) ** 2).mean()
    loss = loss# + latent_loss
    encoder_opt.zero_grad()
    loss.backward()
    encoder_opt.step()
    return loss


def train_decoder(encoder, decoder, decoder_opt):
    batch, true_latents = swirl_data(BS)
    batch = ptu.np_to_var(batch)

    latents = encoder.encode(batch)
    decoder_output = decoder(latents)
    decoder_means = decoder_output[:, 0:2]
    decoder_log_stds = decoder_output[:, 2:4]
    distribution = Normal(decoder_means, decoder_log_stds.exp())
    reconstruction_log_prob = distribution.log_prob(batch).sum(dim=1)

    loss = - reconstruction_log_prob.mean()
    decoder_opt.zero_grad()
    loss.backward()
    decoder_opt.step()
    return loss


def train_alternating(*_):
    encoder = Encoder(
        nn.Linear(2, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, 2),
    )
    encoder_opt = Adam(encoder.parameters())
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
    decoder_opt = Adam(decoder.parameters())

    encoder_losses = []
    decoder_losses = []
    for _ in range(100):
        for _ in range(N_BATCHES):
            encoder_losses.append(
                train_encoder(encoder, decoder, encoder_opt).data.numpy()
            )
        for _ in range(N_BATCHES):
            decoder_losses.append(
                train_decoder(encoder, decoder, decoder_opt).data.numpy()
            )

    # Visualize
    vis_samples_np, true_latents_np = swirl_data(N_VIS)
    vis_samples = ptu.np_to_var(vis_samples_np)
    true_xy_mean_np = t_to_xy(true_latents_np)
    latents = encoder.encode(vis_samples)
    reconstructed_samples = decoder.decode(latents).data.numpy()
    generated_samples = decoder.decode(
        ptu.Variable(torch.randn(*latents.shape))
    ).data.numpy()

    plt.subplot(2, 2, 1)
    plt.plot(np.array(encoder_losses))
    plt.title("Encoder Loss")
    plt.subplot(2, 2, 2)
    plt.plot(np.array(decoder_losses))
    plt.title("Decoder Loss")

    plt.subplot(2, 3, 4)
    plt.plot(generated_samples[:, 0], generated_samples[:, 1], '.')
    plt.title("Generated Samples")
    plt.subplot(2, 3, 5)
    plt.plot(reconstructed_samples[:, 0], reconstructed_samples[:, 1], '.')
    estimated_means = t_to_xy(latents.data.numpy())
    # plt.plot(estimated_means[:, 0], estimated_means[:, 1], '.')
    plt.title("Reconstruction")
    # plt.legend(["Samples", "Projected Latents"])
    plt.subplot(2, 3, 6)
    plt.plot(vis_samples_np[:, 0], vis_samples_np[:, 1], '.')
    plt.plot(true_xy_mean_np[:, 0], true_xy_mean_np[:, 1], '.')
    plt.title("Original Samples")
    plt.legend(["Original", "True means"])
    plt.show()


def train():
    encoder = Encoder(
        nn.Linear(2, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, 2),
    )
    encoder_opt = Adam(encoder.parameters())
    # This is the first place that we cheat. However, this pretraining isn't
    # needed if you just add the loss to the training (see below)
    # pretrain_encoder(encoder, encoder_opt)
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
    decoder_opt = Adam(decoder.parameters())
    print("Done training encoder")

    losses = []
    kls = []
    log_probs = []
    for _ in range(N_BATCHES):
        batch, true_latents = swirl_data(BS)
        batch = ptu.np_to_var(batch)

        latents, means, log_stds, stds = encoder.get_encoding_and_suff_stats(
            batch
        )
        kl = kl_to_prior(means, log_stds, stds)


        latents = encoder.encode(batch)
        # decoder_output = decoder(latents.detach())
        decoder_output = decoder(latents)
        decoder_means = decoder_output[:, 0:2]
        decoder_log_stds = decoder_output[:, 2:4]
        distribution = Normal(decoder_means, decoder_log_stds.exp())
        reconstruction_log_prob = distribution.log_prob(batch).sum(dim=1)

        elbo = - kl + reconstruction_log_prob
        loss = - elbo.mean()
        # This is the second place where we cheat:
        latent_loss = ((ptu.np_to_var(true_latents) - latents) ** 2).mean()
        loss = loss + latent_loss
        decoder_opt.zero_grad()
        encoder_opt.zero_grad()
        loss.backward()
        decoder_opt.step()
        encoder_opt.step()

        losses.append(loss.data.numpy())
        kls.append(kl.mean().data.numpy())
        log_probs.append(reconstruction_log_prob.mean().data.numpy())

    # Visualize
    vis_samples_np, true_latents_np = swirl_data(N_VIS)
    vis_samples = ptu.np_to_var(vis_samples_np)
    true_xy_mean_np = t_to_xy(true_latents_np)
    latents = encoder.encode(vis_samples)
    reconstructed_samples = decoder.decode(latents).data.numpy()
    generated_samples = decoder.decode(
        ptu.Variable(torch.randn(*latents.shape))
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
    estimated_means = t_to_xy(latents.data.numpy())
    plt.plot(estimated_means[:, 0], estimated_means[:, 1], '.')
    plt.title("Reconstruction")
    plt.subplot(2, 3, 6)
    plt.plot(vis_samples_np[:, 0], vis_samples_np[:, 1], '.')
    plt.plot(true_xy_mean_np[:, 0], true_xy_mean_np[:, 1], '.')
    plt.title("Original Samples")
    plt.legend(["Original", "True means"])
    plt.show()


if __name__ == '__main__':
    train_alternating()
    # train()
