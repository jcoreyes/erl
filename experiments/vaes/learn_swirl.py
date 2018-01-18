"""
Train a neural network to learn the swirl function.

Input: xy coordinate
Desired output: t that generated the xy-corrdinate

This is basically doing supervised learning for the encoder in a VAE.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.distributions import Normal
from torch.optim import Adam

SWIRL_RATE = 1
T = 10
BS = 128
N_BATCHES = 1000
N_VIS = 1000
HIDDEN_SIZE = 32
SIGMA = 1. / (T * 2)


def swirl_data(batch_size):
    t = np.random.uniform(size=batch_size, low=0, high=T)
    x = t * np.cos(t * SWIRL_RATE) / T
    y = t * np.sin(t * SWIRL_RATE) / T
    data = np.array([x, y]).T
    noise = np.random.randn(batch_size, 2) * SIGMA
    return data + noise, data, t.reshape(-1, 1)


def t_to_xy(t):
    x = t * np.cos(t * SWIRL_RATE) / T
    y = t * np.sin(t * SWIRL_RATE) / T
    return np.array([x, y]).T


def np_to_var(np_array):
    return Variable(torch.from_numpy(np_array).float())


def train_deterministic():
    network = nn.Sequential(
        nn.Linear(2, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, 1),
    )
    opt = Adam(network.parameters())

    losses = []
    for _ in range(N_BATCHES):
        x_np, _, y_np = swirl_data(BS)
        x = np_to_var(x_np)
        y = np_to_var(y_np)
        y_hat = network(x)
        loss = ((y_hat - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.data.numpy())

    # Visualize
    x_np, x_np_no_noise, y_np = swirl_data(BS)
    x = np_to_var(x_np)
    y_hat = network(x)
    y_hat_np = y_hat.data.numpy()
    x_hat_np = t_to_xy(y_hat_np[:, 0])

    plt.subplot(2, 1, 1)
    plt.plot(np.array(losses))
    plt.title("Training Loss")

    plt.subplot(2, 1, 2)
    plt.plot(x_np[:, 0], x_np[:, 1], '.')
    plt.plot(x_np_no_noise[:, 0], x_np_no_noise[:, 1], '.')
    plt.plot(x_hat_np[:, 0], x_hat_np[:, 1], '.')
    plt.title("Samples")
    plt.legend(["Samples", "No Noise", "Estimates"])
    plt.show()


def output_to_samples(output, std=None):
    y_hat_mean = output[:, 0:1]
    epsilon = Variable(torch.randn(*y_hat_mean.shape))
    if std is None:
        y_hat_log_std = output[:, 1:2]
        return epsilon * y_hat_log_std.exp() + y_hat_mean
    else:
        return epsilon * std + y_hat_mean


def train_stochastic_reparam():
    network = nn.Sequential(
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
    opt = Adam(network.parameters())

    losses = []
    for _ in range(N_BATCHES):
        x_np, _, y_np = swirl_data(BS)
        x = np_to_var(x_np)
        y = np_to_var(y_np)
        y_hat = output_to_samples(network(x))
        loss = ((y_hat - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.data.numpy())

    # Visualize
    x_np, x_np_no_noise, y_np = swirl_data(BS)
    x = np_to_var(x_np)
    y_hat = output_to_samples(network(x))
    y_hat_np = y_hat.data.numpy()
    x_hat_np = t_to_xy(y_hat_np[:, 0])

    plt.subplot(2, 1, 1)
    plt.plot(np.array(losses))
    plt.title("Training Loss")

    plt.subplot(2, 1, 2)
    plt.plot(x_np[:, 0], x_np[:, 1], '.')
    plt.plot(x_np_no_noise[:, 0], x_np_no_noise[:, 1], '.')
    plt.plot(x_hat_np[:, 0], x_hat_np[:, 1], '.')
    plt.title("Samples")
    plt.legend(["Samples", "No Noise", "Estimates"])
    plt.show()


def train_stochastic_score_function():
    network = nn.Sequential(
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
    opt = Adam(network.parameters())

    losses = []
    for _ in range(N_BATCHES):
        x_np, _, y_np = swirl_data(BS)
        x = np_to_var(x_np)
        y = np_to_var(y_np)
        output = network(x)
        y_hat_mean = output[:, 0:1]
        y_hat_log_std = output[:, 1:2]
        y_hat_std = y_hat_log_std.exp()
        y_hat = output_to_samples(output)
        log_prob = Normal(y_hat_mean, y_hat_std.exp()).log_prob(y_hat)
        # y_hat = output_to_samples(output, SIGMA)
        # log_prob = Normal(y_hat_mean, SIGMA).log_prob(y_hat)
        error = ((y_hat - y) ** 2).detach()
        loss = (log_prob * error).mean()
        # This is needed to make it numerically stable...but still, it learns
        # nothing
        reg_loss = (y_hat_mean ** 2 + (y_hat_std)**2).mean()
        loss = loss + reg_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.data.numpy())

    # Visualize
    x_np, x_np_no_noise, y_np = swirl_data(BS)
    x = np_to_var(x_np)
    output = network(x)
    y_hat = output_to_samples(output)
    y_hat_np = y_hat.data.numpy()
    x_hat_np = t_to_xy(y_hat_np[:, 0])

    plt.subplot(2, 1, 1)
    plt.plot(np.array(losses))
    plt.title("Training Loss")

    plt.subplot(2, 1, 2)
    plt.plot(x_np[:, 0], x_np[:, 1], '.')
    plt.plot(x_np_no_noise[:, 0], x_np_no_noise[:, 1], '.')
    plt.plot(x_hat_np[:, 0], x_hat_np[:, 1], '.')
    plt.title("Samples")
    plt.legend(["Samples", "No Noise", "Estimates"])
    plt.show()


if __name__ == '__main__':
    # Works pretty easily
    train_deterministic()
    # Also works pretty easily
    train_stochastic_reparam()
    # Doesn't seem to work
    # train_stochastic_score_function()
