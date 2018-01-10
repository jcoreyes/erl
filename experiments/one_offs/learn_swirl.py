"""
Train a neural network to learn the swirl function
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.optim import Adam

SWIRL_RATE = 1
T = 10


def swirl_data(batch_size):
    t = np.random.uniform(size=batch_size, low=0, high=T)
    x = t * np.cos(t * SWIRL_RATE) / T
    y = t * np.sin(t * SWIRL_RATE) / T
    data = np.array([x, y]).T
    noise = np.random.randn(batch_size, 2) / (T * 2)
    return t.reshape(-1, 1), data + noise, data


def np_to_var(np_array):
    return Variable(torch.from_numpy(np_array).float())


BS = 16
N_BATCHES = 10000
N_VIS = 1000


def train_deterministic():
    network = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    opt = Adam(network.parameters())

    losses = []
    for _ in range(N_BATCHES):
        x_np, y_np, _ = swirl_data(BS)
        x = np_to_var(x_np)
        y = np_to_var(y_np)
        y_hat = network(x)
        loss = ((y_hat - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.data.numpy())

    # Visualize
    x_np, y_np, y_np_no_noise = swirl_data(N_VIS)
    x = np_to_var(x_np)
    y_hat = network(x)
    y_hat_np = y_hat.data.numpy()

    plt.subplot(2, 1, 1)
    plt.plot(np.array(losses))
    plt.title("Training Loss")

    plt.subplot(2, 1, 2)
    plt.plot(y_np[:, 0], y_np[:, 1], '.')
    plt.plot(y_np_no_noise[:, 0], y_np_no_noise[:, 1], '.')
    plt.plot(y_hat_np[:, 0], y_hat_np[:, 1], '.')
    plt.title("Samples")
    plt.legend(["Samples", "No Noise", "Estimates"])
    plt.show()


if __name__ == '__main__':
    train_deterministic()
