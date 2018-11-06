import torch
from railrl.torch.core import PyTorchModule
import numpy as np
import abc
from torch.distributions import Normal
from torch.nn import functional as F

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
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def decode(self, latents):
        """
        :param latents:
        :return: reconstruction, obs_distribution_params
        """
        raise NotImplementedError()

    def logprob(self, inputs, obs_distribution_params):
        """
        :param inputs:
        :param obs_distribution_params:
        :return: log probability of input under decoder
        """
        raise NotImplementedError()

    def kl_divergence(self, latent_distribution_params):
        """
        :param latent_distribution_params:
        :return:
        """
        raise NotImplementedError()

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
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def reparameterize(self, latent_distribution_params):
        if self.training:
            return self.rsample(latent_distribution_params)
        else:
            return latent_distribution_params[0]

    def forward(self, input):
        """

        :param input:
        :return: reconstructed input, latent_distribution_params
        """
        mu, logvar = self.encode(input)
        z = self.reparameterize((mu, logvar))
        reconstructions, obs_distribution_params = self.decode(z)
        return reconstructions, obs_distribution_params, (mu, logvar)

    def vector_kl_divergence(self, latent_distribution_params):
        mu, logvar = latent_distribution_params
        return - torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    def kl_divergence(self, latent_distribution_params):
        return self.vector_kl_divergence(latent_distribution_params).mean()

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

def compute_bernoulli_log_prob(x, recon_x, vector_dimension):
    # Multiply back in the vector_dimension so the cross entropy is only averaged over the batch size
    return -1* F.binary_cross_entropy(
        recon_x,
        x.narrow(start=0, length=vector_dimension,
                 dim=1).contiguous().view(-1, vector_dimension),
        reduction='elementwise_mean',
    ) * vector_dimension

def compute_vectorized_bernoulli_log_prob(x, recon_x, vector_dimension):
    # Multiply back in the vector_dimension so the cross entropy is only averaged over the batch size
    return -1* F.binary_cross_entropy(
        recon_x,
        x.narrow(start=0, length=vector_dimension,
                 dim=1).contiguous().view(-1, vector_dimension),
        reduction='none',
    )

def compute_gaussian_log_prob(input, dec_mu, dec_var, vector_dimension):
    dec_mu = dec_mu.view(-1, vector_dimension)
    dec_var = dec_var.view(-1, vector_dimension)
    decoder_dist = Normal(dec_mu, dec_var.pow(0.5))
    input = input.view(-1, vector_dimension)
    log_probs = decoder_dist.log_prob(input)
    vals = log_probs.sum(dim=1, keepdim=True)
    return vals.mean()

