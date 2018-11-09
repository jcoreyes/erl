import torch
from railrl.torch.core import PyTorchModule
import numpy as np
import abc
from torch.distributions import Normal
from torch.nn import functional as F
from railrl.torch import pytorch_util as ptu

class VAEBase(PyTorchModule,  metaclass=abc.ABCMeta):
    def __init__(
            self,
            representation_size,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.representation_size = representation_size

    @abc.abstractmethod
    def encode(self, input):
        """
        :param input:
        :return: latent_distribution_params
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def rsample(self, latent_distribution_params):
        """

        :param latent_distribution_params:
        :return: latents
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reparameterize(self, latent_distribution_params):
        """

        :param latent_distribution_params:
        :return: latents
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def decode(self, latents):
        """
        :param latents:
        :return: reconstruction, obs_distribution_params
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def logprob(self, inputs, obs_distribution_params):
        """
        :param inputs:
        :param obs_distribution_params:
        :return: log probability of input under decoder
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def kl_divergence(self, latent_distribution_params):
        """
        :param latent_distribution_params:
        :return: kl div between latent_distribution_params and prior on latent space
        """
        raise NotImplementedError()

    def forward(self, input):
        """
        :param input:
        :return: reconstructed input, obs_distribution_params, latent_distribution_params
        """
        latent_distribution_params = self.encode(input)
        latents = self.reparameterize(latent_distribution_params)
        reconstructions, obs_distribution_params = self.decode(latents)
        return reconstructions, obs_distribution_params, latent_distribution_params


class GaussianLatentVAE(VAEBase):
    def __init__(
            self,
            representation_size,
    ):
        self.save_init_params(locals())
        super().__init__(representation_size)
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)

    def rsample(self, latent_distribution_params):
        mu, logvar = latent_distribution_params
        mu = mu.view((mu.size()[0], mu.size()[1]))
        stds = (0.5 * logvar).exp()
        stds = stds.view(stds.size()[0], stds.size()[1])
        epsilon = ptu.randn((mu.size()[0], mu.size()[1]))
        latents = epsilon * stds + mu
        return latents

    def rsample_multiple_latents(self, latent_distribution_params, num_latents_to_sample=1):
        mu, logvar = latent_distribution_params
        mu = mu.view((mu.size()[0], 1, mu.size()[1]))
        stds = (0.5 * logvar).exp()
        stds = stds.view(stds.size()[0], 1, stds.size()[1])
        epsilon = ptu.randn((mu.size()[0], num_latents_to_sample, mu.size()[1]))
        latents = epsilon * stds + mu
        return latents

    def reparameterize(self, latent_distribution_params):
        if self.training:
            return self.rsample(latent_distribution_params)
        else:
            return latent_distribution_params[0]

    def kl_divergence(self, latent_distribution_params):
        mu, logvar = latent_distribution_params
        return - torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def __getstate__(self):
        d = super().__getstate__()
        # Add these explicitly in case they were modified
        d["_dist_mu"] = self.dist_mu
        d["_dist_std"] = self.dist_std
        return d

    def __setstate__(self, d):
        super().__setstate__(d)
        self.dist_mu = d["_dist_mu"]
        self.dist_std = d["_dist_std"]

def compute_bernoulli_log_prob(x, reconstruction_of_x):
    # Multiply back in the vector_dimension so the cross entropy is only averaged over the batch size
    return -1* F.binary_cross_entropy(
        reconstruction_of_x,
        x,
        reduction='elementwise_mean'
    )

def compute_gaussian_log_prob(input, dec_mu, dec_var):
    decoder_dist = Normal(dec_mu, dec_var.pow(0.5))
    log_probs = decoder_dist.log_prob(input)
    vals = log_probs.sum(dim=1, keepdim=True)
    return vals.mean()

