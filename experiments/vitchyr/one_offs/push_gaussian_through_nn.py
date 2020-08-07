"""
Say you have a neural network `f` that takes as input a vector.
Consider the task of estimating

M = E[f(x)]

where x ~ N(0, I)

Some ideas for approximating M:

1. (Naive) \hat M = f(0)
2. (Monte Carlo) \hat M = \frac{1}{N} \sum_i^N f(x_i), x_i ~ N(0, I)
3. (Convolution) \hat M = \hat f (0)

where \hat f is a convolved version of `f`

For example, if f is a neural network with tanh activations, \hat f is the same
network but with tanh(x) replaced by $\tanh(x /\sqrt{1 + \pi / 2})$.
j
According to
    https://arxiv.org/pdf/1601.04114.pdf
given the activation on the left, the corresponding convolved function is

    tanh(x)     tanh(x / \sqrt{1 + \pi/2 \sigma^2)
    relu(x)     \sigma / \sqrt{2 \pi} e^{-x^2 / (2\sigma^2)}
                     + x/2 * (1 + erf(x / (\sqrt{2} * \sigma))

See

https://arxiv.org/pdf/1602.05610.pdf
"""
from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.optim import Adam

from rlkit.torch.networks import Mlp

INPUT_SIZE = 10
OUTPUT_SIZE = 1
N = 10
N_TRUTH = 100000


def estimate_mean_naive(network):
    return network.eval_np(np.zeros((1, INPUT_SIZE)))[0, 0]


def estimate_mean_monte_carlo(network, num_samples):
    samples = torch.randn(num_samples, INPUT_SIZE)
    res = network(Variable(samples)).data.numpy()
    return np.mean(res, axis=0)


def estimate_mean_convolution(network):
    # No need to input bias! That's because the network already accounts for
    # that (duh)
    return network.eval_np(np.zeros((1, INPUT_SIZE)), convolve=True)[0, 0]


class TanhNetwork(Mlp):
    def forward(self, input, convolve=False):
        h = input
        for i, fc in enumerate(self.fcs):
            if convolve:
                variance = torch.sum(fc.weight * fc.weight, dim=1).unsqueeze(0)
                conv_factor_inv = torch.sqrt(1 + np.pi / 2 * variance)
                h = torch.tanh(fc(h) / conv_factor_inv)
            else:
                h = torch.tanh(fc(h))
        return self.last_fc(h)


def train(network):
    optimizer = Adam(network.parameters())
    loss_fctn = nn.MSELoss()
    for _ in range(100):
        x = torch.rand(64, 1) * 10 - 5
        y = torch.sin(x)

        x = Variable(x)
        y = Variable(y)
        y_hat = network(x)
        loss = loss_fctn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return network


def print_estimates(network):
    truth = estimate_mean_monte_carlo(network, 100000)
    naive = estimate_mean_naive(network)
    mc_1 = estimate_mean_monte_carlo(network, 1)
    mc_10 = estimate_mean_monte_carlo(network, 10)
    mc_100 = estimate_mean_monte_carlo(network, 100)
    conv = estimate_mean_convolution(network)
    print("naive", naive)
    print("mc_1", mc_1)
    print("mc_10", mc_10)
    print("mc_100", mc_100)
    print("conv", conv)
    print("truth", truth)


def run_exp(network_generator):
    name_to_errors = OrderedDict()
    name_to_errors["truth"] = []
    estim_mc_1 = lambda x: estimate_mean_monte_carlo(x, 1)
    estim_mc_10 = lambda x: estimate_mean_monte_carlo(x, 10)
    estim_mc_100 = lambda x: estimate_mean_monte_carlo(x, 100)
    for _ in range(N):
        network = network_generator()
        truth = estimate_mean_monte_carlo(network, N_TRUTH)
        for name, estimator in [
            ("MC 1", estim_mc_1),
            ("MC 10", estim_mc_10),
            ("MC 100", estim_mc_100),
            ("Naive", estimate_mean_naive),
            ("Convolve", estimate_mean_convolution),
        ]:
            estim = estimator(network)
            name_to_errors.setdefault(name, []).append(truth - estim)

    # See how reliable "truth" is
    network = network_generator()
    for _ in range(N):
        truth = estimate_mean_monte_carlo(network, N_TRUTH)
        name_to_errors["truth"].append(truth)

    for name, errors in name_to_errors.items():
        print("{0}: {1} ({2})".format(
            name, np.mean(errors, axis=0), np.std(errors, axis=0) / np.sqrt(N)
        ))
    for name, errors in name_to_errors.items():
        print("{0}: {1} ({2})".format(
            name, np.mean(errors), np.std(errors) / np.sqrt(N)
        ))


def get_network_generator(params, train_net=False):
    def network_generator():
        tanh_net = TanhNetwork(**params)
        if train_net:
            train(tanh_net)
        return tanh_net

    return network_generator


def main():
    params = dict(
        hidden_sizes=[10],
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        init_w=1,
    )
    print("Tanh")
    run_exp(get_network_generator(
        params,
        # train_net=True,
        train_net=False,
    ))


if __name__ == '__main__':
    main()
