from __future__ import print_function
import torch
import numpy as np
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from railrl.pythonplusplus import identity
from torch.autograd import Variable
from railrl.torch import pytorch_util as ptu
from railrl.torch.networks import ConcatMlp, TanhMlpPolicy, MlpPolicy
from railrl.torch.networks import CNN, TwoHeadDCNN, DCNN
from railrl.torch.vae.vae_base import compute_bernoulli_log_prob, compute_gaussian_log_prob, GaussianLatentVAE
from railrl.torch.vae.conv_vae import ConvVAE
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(

            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)


    def forward(self, inputs, ):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)

class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, discrete_size):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                    out_channels=embedding_dim,
                                    kernel_size=1,
                                    stride=1)

    def forward(self, inputs, ):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens//2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, gaussion_prior=False):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(
                self._num_embeddings, self._embedding_dim)
        
        if gaussion_prior:
            self._embedding.weight.data.normal_()

        else:
            self._embedding.weight.data.uniform_(
                -1/self._num_embeddings, 1/self._num_embeddings)
        
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(
            encodings, self._embedding.weight).view(input_shape)

        # Loss
        #e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='sum')
        #q_latent_loss = F.mse_loss(quantized, inputs.detach(), reduction='sum')
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                                          torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(
            num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(
            encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        #e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='sum')
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                                          torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices


class VQ_VAE(nn.Module):
    def __init__(
        self,
        embedding_dim=3,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        imsize=48,
        decay=0.0):
        super(VQ_VAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings
        self._encoder = Encoder(input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(self.embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        
        #Calculate latent sizes
        if imsize == 48:
            self.root_len = 12
        elif imsize == 84:
            self.root_len = 21
        else:
            raise ValueError(imsize)

        self.discrete_size = self.root_len * self.root_len
        self.representation_size = self.discrete_size * self.embedding_dim
        #Calculate latent sizes

    def compute_loss(self, inputs):
        inputs = inputs.view(-1,
                            self.input_channels,
                            self.imsize,
                            self.imsize)

        vq_loss, quantized, perplexity, _ = self.quantize_image(inputs)
        x_recon = self.decode(quantized)
        
        recon_error = F.mse_loss(x_recon, inputs)
        return vq_loss, quantized, x_recon, perplexity, recon_error

    def quantize_image(self, inputs):
        inputs = inputs.view(-1,
                            self.input_channels,
                            self.imsize,
                            self.imsize)

        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)
        return self._vq_vae(z)

    def encode(self, inputs, cont=True):
        _, quantized, _, encodings = self.quantize_image(inputs)

        if cont:
            return quantized.reshape(-1, self.representation_size)
        return encodings.reshape(-1, self.discrete_size)

    def latent_to_square(self, latents):
        return latents.reshape(-1, self.root_len, self.root_len)

    def discrete_to_cont(self, e_indices):
        e_indices = self.latent_to_square(e_indices)
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)

        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)

        e_weights = self._vq_vae._embedding.weight
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)

        z_q = torch.matmul(min_encodings, e_weights).view(input_shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn

    def decode(self, latents, cont=True):
        if cont:
            z_q = latents.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
        else:
            z_q = self.discrete_to_cont(latents)

        return self._decoder(z_q)

    def encode_one_np(self, inputs, cont=True):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))[0]

    def encode_np(self, inputs, cont=True):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))

    def decode_one_np(self, inputs, cont=True):
        return np.clip(ptu.get_numpy(self.decode(ptu.from_numpy(inputs).reshape(1, -1), cont=cont))[0], 0, 1)


    def decode_np(self, inputs, cont=True):
        return np.clip(ptu.get_numpy(self.decode(ptu.from_numpy(inputs), cont=cont)), 0, 1)

# class VQ_VAE(nn.Module):
#     def __init__(
#         self,
#         embedding_dim,
#         root_len=15,
#         input_channels=3,
#         num_hiddens=128,
#         num_residual_layers=3,
#         num_residual_hiddens=64,
#         num_embeddings=512,
#         decoder_output_activation=None, #IGNORED FOR NOW
#         architecture=None, #IGNORED FOR NOW
#         min_variance=1e-3,
#         commitment_cost=0.25,
#         imsize=48,
#         decay=0.0,
#         ):
#         super(VQ_VAE, self).__init__()
#         self.log_min_variance = float(np.log(min_variance))
#         self.imsize = imsize
#         self.embedding_dim = embedding_dim
#         self.input_channels = input_channels
#         self.imlength = imsize * imsize * input_channels
        
#         self._encoder = Encoder(input_channels, num_hiddens,
#                                 num_residual_layers,
#                                 num_residual_hiddens)
#         self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
#                                       out_channels=self.embedding_dim,
#                                       kernel_size=1,
#                                       stride=1)

#         self._decoder = Decoder(self.embedding_dim,
#                                 num_hiddens,
#                                 num_residual_layers,
#                                 num_residual_hiddens)

#         if decay > 0.0:
#             self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
#                                               commitment_cost, decay)
#         else:
#             self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
#                                            commitment_cost)
        
#         #Calculate latent sizes
#         if imsize == 48:
#             self.root_conv_size = 12
#         elif imsize == 84:
#             self.root_conv_size = 21
#         else:
#             raise ValueError(imsize)

#         self.conv_size = self.root_conv_size * self.root_conv_size * self.embedding_dim
#         self.root_len = root_len
#         self.discrete_size = root_len * root_len
#         self.representation_size = self.discrete_size * self.embedding_dim
#         #Calculate latent sizes

#         assert self.representation_size <= self.conv_size  # This is a bad idea (wrong bottleneck)

#         self.f_enc = nn.Linear(self.conv_size, self.representation_size)
#         self.f_dec = nn.Linear(self.representation_size, self.conv_size)

#         self.f_enc.weight.data.uniform_(-1e-3, 1e-3)
#         self.f_enc.bias.data.uniform_(-1e-3, 1e-3)
#         self.f_dec.weight.data.uniform_(-1e-3, 1e-3)
#         self.f_dec.bias.data.uniform_(-1e-3, 1e-3)

#     def compute_loss(self, obs):
#         obs = obs.view(-1,
#           self.input_channels,
#           self.imsize,
#           self.imsize)

#         vq_loss, quantized, perplexity, _ = self.encode_image(obs)

#         recon = self.decode(quantized)
#         recon_error = F.mse_loss(recon, obs)

#         return vq_loss, quantized, recon, perplexity, recon_error


#     def encode_image(self, obs):
#         obs = obs.view(-1,
#           self.input_channels,
#           self.imsize,
#           self.imsize)

#         z_conv = self._encoder(obs)
#         z_conv = self._pre_vq_conv(z_conv)

#         return self.compress(z_conv)

#     def compress(self, z_conv):
#         z_conv = z_conv.view(-1, self.conv_size)
#         z = self.f_enc(z_conv).view(-1, self.embedding_dim, self.root_len, self.root_len)
#         vq_loss, quantized, perplexity, encodings = self._vq_vae(z)
#         quantized = quantized.view(-1, self.representation_size)
        
#         return vq_loss, quantized, perplexity, encodings

#     def decompress(self, quantized):
#         z_conv = self.f_dec(quantized)
#         z_conv = z_conv.view(-1, self.embedding_dim, self.root_conv_size, self.root_conv_size)   
#         return z_conv


#     def encode(self, inputs, cont=True):
#         _, quantized, _, encodings = self.encode_image(inputs)

#         if cont:
#           return quantized.view(-1, self.representation_size)
#         return encodings.view(-1, self.discrete_size)

#     def sample_prior(self, batch_size):
#         z_s = ptu.randn(batch_size, self.representation_size)
#         return z_s

#     def decode(self, latents, cont=True):
#         if not cont:
#           latents = self.discrete_to_cont(latents)

#         z_conv = self.decompress(latents)

#         return self._decoder(z_conv)

#     def discrete_to_cont(self, e_indices):
#         e_indices = e_indices.reshape(-1, self.root_len, self.root_len)
#         input_shape = e_indices.shape + (self.embedding_dim,)
#         e_indices = e_indices.reshape(-1).unsqueeze(1)

#         min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
#         min_encodings.scatter_(1, e_indices, 1)

#         e_weights = self._vq_vae._embedding.weight
#         quantized = torch.matmul(
#             min_encodings, e_weights).view(input_shape)

#         z_q = torch.matmul(min_encodings, e_weights).view(input_shape)
#         z_q = z_q.permute(0, 3, 1, 2).contiguous()
#         return z_q

class CVQVAE(nn.Module):
    def __init__(
        self,
        embedding_dim,
        root_len=21,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        min_variance=1e-3,
        commitment_cost=0.25,
        imsize=48,
        decay=0.0,
        ):
        super(CVQVAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        
        self._encoder = Encoder(2 * input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._cond_encoder = Encoder(input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=1,
                                      kernel_size=1,
                                      stride=1)
        self._cond_pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)

        self._decoder = Decoder(1 + self.embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)
        
        #Calculate latent sizes
        if imsize == 48:
            self.root_conv_size = 12
        elif imsize == 84:
            self.root_conv_size = 21
        else:
            raise ValueError(imsize)

        self.root_len = root_len
        self.conv_size = self.root_conv_size * self.root_conv_size
        self.discrete_size = root_len * root_len
        self.latent_sizes = [self.discrete_size * self.embedding_dim, self.conv_size * self.embedding_dim]
        self.representation_size = sum(self.latent_sizes)
        #Calculate latent sizes

        assert self.latent_sizes[0] <= self.conv_size * self.embedding_dim  # This is a bad idea (wrong bottleneck)

        self.f_enc = nn.Linear(self.conv_size, self.latent_sizes[0])
        self.f_dec = nn.Linear(self.latent_sizes[0], self.conv_size)

        self.f_enc.weight.data.uniform_(-1e-3, 1e-3)
        self.f_enc.bias.data.uniform_(-1e-3, 1e-3)
        self.f_dec.weight.data.uniform_(-1e-3, 1e-3)
        self.f_dec.bias.data.uniform_(-1e-3, 1e-3)

    def compute_loss(self, x_delta, x_cond):
        x_delta = x_delta.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)

        vq_loss, quantized, perplexity, _ = self.encode_images(x_delta, x_cond)

        recon = self.decode(quantized)
        recon_error = F.mse_loss(recon, x_delta)

        return vq_loss, quantized, recon, perplexity, recon_error


    def encode_images(self, x_delta, x_cond):
        x_delta = x_delta.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)

        x_cond = x_cond.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)
        
        x_delta = torch.cat([x_delta, x_cond], dim=1)
        
        z_delta = self._encoder(x_delta)
        z_delta = self._pre_vq_conv(z_delta)

        z_cond = self._cond_encoder(x_cond)
        z_cond = self._cond_pre_vq_conv(z_cond)

        return self.compress(z_delta, z_cond)

    def compress(self, z_delta, z_cond):
        z_delta = z_delta.view(-1, self.conv_size)
        z_cond = z_cond.view(-1, self.conv_size * self.embedding_dim)
        
        z_delta = self.f_enc(z_delta)
        z_delta = z_delta.view(-1, self.embedding_dim, self.root_len, self.root_len)
        
        vq_loss, quantized, perplexity, encodings = self._vq_vae(z_delta)
        quantized = quantized.view(-1, self.latent_sizes[0])

        quantized = torch.cat([quantized, z_cond], dim=1)
        
        return vq_loss, quantized, perplexity, encodings

    def decompress(self, quantized):
        z_delta = quantized[:, :self.latent_sizes[0]]
        z_cond = quantized[:, self.latent_sizes[0]:]

        z_delta = self.f_dec(z_delta)
        z_delta = z_delta.view(-1, 1, self.root_conv_size, self.root_conv_size)
        z_cond = z_cond.view(-1, self.embedding_dim, self.root_conv_size, self.root_conv_size)

        z_conv = torch.cat([z_delta, z_cond], dim=1)
        return z_conv


    def encode(self, x_delta, x_cond, cont=True):
        batch_size = x_delta.shape[0]
        _, quantized, _, encodings = self.encode_images(x_delta, x_cond)

        if cont:
          return quantized.view(batch_size, -1)
        return encodings.view(batch_size, -1)

    def sample_prior(self, batch_size):
        z_s = ptu.randn(batch_size, self.latent_sizes[0])
        return z_s

    def decode(self, latents, cont=True):
        if not cont:
          return 1/0
          latents = self.discrete_to_cont(latents)

        z_conv = self.decompress(latents)

        return self._decoder(z_conv)

    def discrete_to_cont(self, e_indices):
        e_indices = e_indices.reshape(-1, self.root_len, self.root_len)
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)

        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)

        e_weights = self._vq_vae._embedding.weight
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)

        z_q = torch.matmul(min_encodings, e_weights).view(input_shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


class VQ_VAE1(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        imsize=48,
        decay=0.0):
        super(VQ_VAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings
        self._encoder = Encoder(input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(self.embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        
        #Calculate latent sizes
        if imsize == 48: self.root_len = 12
        elif imsize == 84: self.root_len = 21
        else: raise ValueError(imsize)

        self.discrete_size = self.root_len * self.root_len
        self.representation_size = self.discrete_size * self.embedding_dim
        #Calculate latent sizes

    def compute_loss(self, inputs):
        inputs = inputs.view(-1,
                            self.input_channels,
                            self.imsize,
                            self.imsize)
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)
        vq_loss, quantized, perplexity, _ = self._vq_vae(z)

        x_recon = self._decoder(quantized)
        recon_error = F.mse_loss(x_recon, inputs)
        return vq_loss, quantized, x_recon, perplexity, recon_error


    def latent_to_square(self, latents):
        #root_len = int(latents.shape[1] ** 0.5)
        return latents.reshape(-1, self.root_len, self.root_len)

    def encode(self, inputs, cont=True):
        inputs = inputs.view(-1,
                            self.input_channels,
                            self.imsize,
                            self.imsize)
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)
        _, quantized, _, encodings = self._vq_vae(z)

        if cont:
            return quantized.reshape(inputs.shape[0], -1)

        return encodings.reshape(inputs.shape[0], -1)


    def encode_one_np(self, inputs, cont=True):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))[0]


    def encode_np(self, inputs, cont=True):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))

    def decode_one_np(self, inputs, cont=True):
        return np.clip(ptu.get_numpy(self.decode(ptu.from_numpy(inputs).reshape(1, -1), cont=cont))[0], 0, 1)


    def decode_np(self, inputs, cont=True):
        return np.clip(ptu.get_numpy(self.decode(ptu.from_numpy(inputs), cont=cont)), 0, 1)


    def discrete_to_cont(self, e_indices):
        e_indices = self.latent_to_square(e_indices)
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)

        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)

        e_weights = self._vq_vae._embedding.weight
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)

        z_q = torch.matmul(min_encodings, e_weights).view(input_shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn


    def decode(self, latents, cont=True):
        z_q = None
        if cont:
            z_q = latents.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
        else:
            z_q = self.discrete_to_cont(latents)

        return self._decoder(z_q)


    def get_distance(self, s_indices, g_indices):
        assert s_indices.shape == g_indices.shape
        batch_size = s_indices.shape[0]
        s_q = self.discrete_to_cont(s_indices).reshape(batch_size, -1)
        g_q = self.discrete_to_cont(g_indices).reshape(batch_size, -1)
        return ptu.get_numpy(torch.norm(s_q - g_q, dim=1))

    def sample_prior(self, batch_size, cont=True):
        e_indices = self.pixel_cnn.generate(shape=(self.root_len, elf.root_len), batch_size=batch_size)
        e_indices = e_indices.reshape(batch_size, -1)
        if cont:
            return self.discrete_to_cont(e_indices)
        return e_indices

    def logprob(self, images, cont=True):
        batch_size = images.shape[0]
        root_len = int((self.representation_size // self.embedding_dim)**0.5)
        e_indices = self.encode(images, cont=False)
        e_indices = e_indices.reshape(batch_size, root_len, root_len)
        cond = ptu.from_numpy(np.ones((images.shape[0], 1)))
        logits = self.pixel_cnn(e_indices, cond)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        criterion = nn.CrossEntropyLoss(reduction='none')#.cuda()

        logprob = - criterion(
            logits.view(-1, self.num_embeddings),
            e_indices.contiguous().view(-1))

        logprob = logprob.reshape(batch_size, -1).mean(dim=1)

        return logprob


class CVQVAENormal(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        imsize=48,
        decay=0):
        super(CVQVAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings

        #Calculate latent sizes
        if imsize == 48: self.root_len = 12
        elif imsize == 84: self.root_len = 21
        else: raise ValueError(imsize)

        self.discrete_size = self.root_len * self.root_len
        self.latent_sizes = [self.discrete_size * self.embedding_dim, self.discrete_size * self.embedding_dim]
        self.representation_size = sum(self.latent_sizes)
        #Calculate latent sizes

        self._encoder = Encoder(input_channels * 2, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.cond_encoder = Encoder(input_channels, num_hiddens,
                                    num_residual_layers,
                                    num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)

        self.cond_pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
            self.cond_vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)

            self.cond_vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                              commitment_cost)

        self._decoder = Decoder(self.embedding_dim * 2,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.cond_decoder = Decoder(self.embedding_dim,
                                    num_hiddens,
                                    num_residual_layers,
                                    num_residual_hiddens)

    def compute_loss(self, obs, cond):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)

        vq_loss, quantized, perplexity, _ = self._vq_vae(z)
        cond_vq_loss, cond_quantized, cond_perplexity, _ = self.cond_vq_vae(z_cond)
        cat_quantized = torch.cat([quantized, cond_quantized], dim=1)
        
        cond_recon = self.cond_decoder(cond_quantized)
        x_recon = self._decoder(cat_quantized)
        vq_losses = [vq_loss, cond_vq_loss]
        perplexities = [perplexity, cond_perplexity]
        recons = [x_recon, cond_recon]
        errors = [F.mse_loss(x_recon, obs), F.mse_loss(cond_recon, cond)]
        #errors = [F.mse_loss(x_recon, obs), F.mse_loss(cond_recon, cond)]
        return vq_losses, perplexities, recons, errors


    def latent_to_square(self, latents):
        latents = latents.reshape(-1, 2, self.root_len, self.root_len)
        return latents[:, 0], latents[:, 1]

    def encode(self, obs, cond, cont=True):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)

        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)

        _, quantized, _, encodings = self._vq_vae(z)
        _, cond_quantized, _, cond_encodings = self.cond_vq_vae(z_cond)

        if cont:
            z, z_c = quantized, cond_quantized
        else:
            z, z_c = encodings, cond_encodings
        

        z = z.reshape(obs.shape[0], -1)
        z_c = z_c.reshape(cond.shape[0], -1)
        z_cond = torch.cat([z, z_c], dim=1)
        return z_cond


    def conditioned_discrete_to_cont(self, e_indices):
        z_ind, cond_ind = self.latent_to_square(e_indices)
        z = self.discrete_to_cont(z_ind, self._vq_vae._embedding.weight)
        z_cond = self.discrete_to_cont(cond_ind, self.cond_vq_vae._embedding.weight)
        cat_quantized = torch.cat([z, z_cond], dim=1)
        return cat_quantized


    def discrete_to_cont(self, e_indices, e_weights):
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)
        
        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)
        
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)
        
        z_q = torch.matmul(min_encodings, e_weights).view(input_shape) 
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn

    def decode(self, latents, cont=True):
        if cont:
            z = latents.reshape(-1, 2 * self.embedding_dim, self.root_len, self.root_len)
        else:
            z = self.conditioned_discrete_to_cont(latents)
        return self._decoder(z)


    # def sample_prior(self, batch_size, cont=True):
    #     size = self.latent_sizes[0]**0.5
    #     e_indices = self.pixel_cnn.generate(shape=(size, size), batch_size=batch_size)
    #     e_indices = e_indices.reshape(batch_size, -1)
    #     if cont:
    #         return self.discrete_to_cont(e_indices)
    #     return e_indices


class CVQVAE1(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        imsize=48,
        decay=0):
        super(CVQVAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings
        self._encoder = Encoder(input_channels * 2, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.cond_encoder = Encoder(input_channels, num_hiddens,
                                    num_residual_layers,
                                    num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)

        self.cond_pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
            self.cond_vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)

            self.cond_vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                              commitment_cost)

        self._decoder = Decoder(self.embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.representation_size = 0
        self.latent_sizes = []
        self.discrete_size = 0
        self.root_len = 0

    def compute_loss(self, obs, cond):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)

        vq_loss, quantized, perplexity, _ = self._vq_vae(z)
        cond_vq_loss, cond_quantized, cond_perplexity, _ = self.cond_vq_vae(z_cond)
        comb_quantized = quantized + cond_quantized
        #cat_quantized = torch.cat([quantized, cond_quantized], dim=1)

        if self.representation_size == 0:
            z_size = quantized[0].flatten().shape[0]
            z_cond_size = cond_quantized[0].flatten().shape[0]
            self.latent_sizes = [z_size, z_cond_size]
            self.representation_size = z_size + z_cond_size
            self.discrete_size = self.representation_size // self.embedding_dim
            self.root_len = int((self.discrete_size // 2) ** 0.5)
        
        cond_recon = self._decoder(cond_quantized)
        x_recon = self._decoder(comb_quantized)
        vq_losses = [vq_loss, cond_vq_loss]
        perplexities = [perplexity, cond_perplexity]
        recons = [x_recon, cond_recon]
        errors = [F.mse_loss(x_recon, obs), F.mse_loss(cond_recon, cond)]
        return vq_losses, perplexities, recons, errors


    def latent_to_square(self, latents):
        latents = latents.reshape(-1, 2, self.root_len, self.root_len)
        return latents[:, 0], latents[:, 1]

    def encode(self, obs, cond, cont=True):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)

        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)

        _, quantized, _, encodings = self._vq_vae(z)
        _, cond_quantized, _, cond_encodings = self.cond_vq_vae(z_cond)

        if cont:
            z, z_c = quantized, cond_quantized
        else:
            z, z_c = encodings, cond_encodings
        

        z = z.reshape(obs.shape[0], -1)
        z_c = z_c.reshape(cond.shape[0], -1)
        z_cond = torch.cat([z, z_c], dim=1)
        return z_cond


    def conditioned_discrete_to_cont(self, e_indices):
        z_ind, cond_ind = self.latent_to_square(e_indices)
        z = self.discrete_to_cont(z_ind, self._vq_vae._embedding.weight)
        z_cond = self.discrete_to_cont(cond_ind, self.cond_vq_vae._embedding.weight)
        cat_quantized = torch.cat([z, z_cond], dim=1)
        return cat_quantized


    def discrete_to_cont(self, e_indices, e_weights):
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)
        
        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)
        
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)
        
        z_q = torch.matmul(min_encodings, e_weights).view(input_shape) 
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn

    def decode(self, latents, cont=True):
        if cont:
            z = latents.reshape(-1, 2 * self.embedding_dim, self.root_len, self.root_len)
        else:
            z_comb = self.conditioned_discrete_to_cont(latents)
            z_pos = z_comb[:, :self.embedding_dim]
            z_obj = z_comb[:, self.embedding_dim:]
            z = z_pos + z_obj
        return self._decoder(z)



class CVQVAE1(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        min_variance=1e-3,
        imsize=48,
        decay=0):
        super(CVQVAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings
        self.log_min_variance = float(np.log(min_variance))

        #Calculate latent sizes
        if imsize == 48: self.root_len = 12
        elif imsize == 84: self.root_len = 21
        else: raise ValueError(imsize)

        self.discrete_size = self.root_len * self.root_len
        self.latent_sizes = [self.discrete_size, self.discrete_size * self.embedding_dim]
        self.representation_size = sum(self.latent_sizes)
        #Calculate latent sizes

        self._encoder = Encoder(input_channels * 2, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.cond_encoder = Encoder(input_channels, num_hiddens,
                                    num_residual_layers,
                                    num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=1,
                                      #out_channels=self.embedding_dim
                                      kernel_size=1,
                                      stride=1)

        self.cond_pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
            self.cond_vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost, gaussion_prior=True)

            self.cond_vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                              commitment_cost)

        self._decoder = Decoder(self.embedding_dim + 1,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.cond_decoder = Decoder(self.embedding_dim,
                                    num_hiddens,
                                    num_residual_layers,
                                    num_residual_hiddens)

        in_layers = 2
        out_layers = 2
        self.fc_in = nn.ModuleList([nn.Linear(self.discrete_size, self.discrete_size) for i in range(in_layers)])
        self.bn_in = nn.ModuleList([nn.BatchNorm1d(self.discrete_size) for i in range(in_layers)])
        self.fc_out = nn.ModuleList([nn.Linear(self.representation_size, self.representation_size) for i in range(out_layers)])
        self.bn_out = nn.ModuleList([nn.BatchNorm1d(self.representation_size) for i in range(out_layers)])
        self.dropout = nn.Dropout(0.5)
        for f in self.fc_in:
            f.weight.data.uniform_(-1e-3, 1e-3)
            f.bias.data.uniform_(-1e-3, 1e-3)
        for f in self.fc_out:
            f.weight.data.uniform_(-1e-3, 1e-3)
            f.bias.data.uniform_(-1e-3, 1e-3)
        for f in self.bn_in:
            f.weight.data.uniform_(-1e-3, 1e-3)
            f.bias.data.uniform_(-1e-3, 1e-3)
        for f in self.bn_out:
            f.weight.data.uniform_(-1e-3, 1e-3)
            f.bias.data.uniform_(-1e-3, 1e-3)

        self.logvar = nn.Parameter(torch.randn(144))

    def compute_loss(self, obs, cond):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z).reshape(-1, self.discrete_size)
        for i in range(len(self.fc_in)):
            z = self.fc_in[i](self.dropout(self.bn_in[i](F.relu(z))))
        z = z.reshape(-1, 1, self.root_len, self.root_len)
        
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)

        vq_loss, quantized, perplexity, _ = self._vq_vae(z.detach()) 
        #NOTE DETACHED ABOVE
        
        #quantized = self.reparameterize(quantized)
        quantized = self.reparameterize(z)
        
        cond_vq_loss, cond_quantized, cond_perplexity, _ = self.cond_vq_vae(z_cond)
        cat_quantized = torch.cat([quantized, cond_quantized], dim=1).reshape(-1, self.representation_size)
        
        for i in range(len(self.fc_out)):
            cat_quantized = self.fc_out[i](self.dropout(self.bn_out[i]((F.relu(cat_quantized)))))
        cat_quantized = cat_quantized.reshape(-1, self.embedding_dim + 1, self.root_len, self.root_len)

        kle = self.kl_divergence(z)
        cond_recon = self.cond_decoder(cond_quantized)
        x_recon = self._decoder(cat_quantized)
        vq_losses = [vq_loss, cond_vq_loss]
        perplexities = [perplexity, cond_perplexity]
        recons = [x_recon, cond_recon]
        #errors = [F.mse_loss(x_recon, obs), F.mse_loss(cond_recon, cond)]
        errors = [F.mse_loss(x_recon, obs, reduction='sum'), F.mse_loss(cond_recon, cond, reduction='sum')]
        return vq_losses, perplexities, recons, errors, kle


    def latent_to_square(self, latents):
        latents = latents.reshape(-1, 2, self.root_len, self.root_len)
        return latents[:, 0], latents[:, 1]

    def encode(self, obs, cond, cont=True):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z).reshape(-1, self.discrete_size)        
        for i in range(len(self.fc_in)):
            z = self.fc_in[i](self.dropout(self.bn_in[i](F.relu(z))))
        z = z.reshape(-1, 1, self.root_len, self.root_len)

        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)

        _, quantized, _, encodings = self._vq_vae(z)
        _, cond_quantized, _, cond_encodings = self.cond_vq_vae(z_cond)
        quantized = self.reparameterize(z)

        if cont:
            #z, z_c = z, cond_quantized
            z, z_c = quantized, cond_quantized
        else:
            z, z_c = encodings, cond_encodings
        

        z = z.reshape(obs.shape[0], -1)
        z_c = z_c.reshape(cond.shape[0], -1)
        z_cond = torch.cat([z, z_c], dim=1)
        return z_cond

    def reparameterize(self, quantized):
        if self.training:
            return self.rsample(quantized)
        return quantized

    def kl_divergence(self, quantized):
        logvar = self.log_min_variance + torch.abs(self.logvar)
        mu = quantized.reshape(-1, self.latent_sizes[0])
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def rsample(self, mu):
        logvar = self.log_min_variance + torch.abs(self.logvar)
        stds = (0.5 * logvar).exp()
        stds = stds.repeat(mu.shape[0], 1).reshape(*mu.size())
        epsilon = ptu.randn(*mu.size())
        latents = epsilon * stds + mu
        return latents
    
    # def rsample(self, mu):
    #     logvar = self.log_min_variance + torch.abs(self.logvar)
    #     stds = (0.5 * logvar).exp()
    #     epsilon = ptu.randn(*mu.size())
    #     latents = epsilon * stds + mu
    #     return latents

    def conditioned_discrete_to_cont(self, e_indices):
        z_ind, cond_ind = self.latent_to_square(e_indices)
        z = self.discrete_to_cont(z_ind, self._vq_vae._embedding.weight)
        z_cond = self.discrete_to_cont(cond_ind, self.cond_vq_vae._embedding.weight)
        cat_quantized = torch.cat([z, z_cond], dim=1)
        return cat_quantized


    def discrete_to_cont(self, e_indices, e_weights):
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)
        
        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)
        
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)
        
        z_q = torch.matmul(min_encodings, e_weights).view(input_shape) 
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn

    def decode(self, latents, cont=True):
        if cont:
            for i in range(len(self.fc_out)):
                latents = self.fc_out[i](self.dropout(self.bn_out[i](F.relu(latents))))
            z = latents.reshape(-1, self.embedding_dim + 1, self.root_len, self.root_len)
            #z = latents.reshape(-1, 2 * self.embedding_dim, self.root_len, self.root_len)
        else:
            z = self.conditioned_discrete_to_cont(latents)
        return self._decoder(z)

    def sample_prior(self, batch_size, cond, cont=True):
        if cond.shape[0] == 1:
            cond = cond.repeat(batch_size, 1)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        _, cond_quantized, _, cond_encodings = self.cond_vq_vae(z_cond)

        z = ptu.randn(batch_size, 1, self.root_len, self.root_len)
        _, quantized, _, encodings = self._vq_vae(z)

        if cont:
            z, z_c = z, cond_quantized
            #z, z_c = quantized, cond_quantized
        else:
            z, z_c = encodings, cond_encodings

        z = z.reshape(batch_size, -1)
        z_c = z_c.reshape(batch_size, -1)
        z_cond = torch.cat([z, z_c], dim=1)
        return z_cond

class CVQVAE2(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        min_variance=1e-3,
        imsize=48,
        decay=0):
        super(CVQVAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings
        self.log_min_variance = float(np.log(min_variance))

        #Calculate latent sizes
        if imsize == 48: self.root_len = 12
        elif imsize == 84: self.root_len = 21
        else: raise ValueError(imsize)

        self.discrete_size = self.root_len * self.root_len
        self.latent_sizes = [self.discrete_size * self.embedding_dim, self.discrete_size * self.embedding_dim]
        self.representation_size = sum(self.latent_sizes)
        #Calculate latent sizes

        self._encoder = Encoder(input_channels * 2, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self.cond_encoder = Encoder(input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self.cond_pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
            self.cond_vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)

            self.cond_vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                              commitment_cost)

        self._decoder = Decoder(self.embedding_dim * 2,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.cond_decoder = Decoder(self.embedding_dim,
                                    num_hiddens,
                                    num_residual_layers,
                                    num_residual_hiddens)


        self.f_mu = nn.Linear(self.latent_sizes[0], self.latent_sizes[0])
        self.f_logvar = nn.Linear(self.latent_sizes[0], self.latent_sizes[0])
        self.f_lambda = nn.Linear(self.latent_sizes[1], self.embedding_dim)
        self.f_beta = nn.Linear(self.latent_sizes[1], self.embedding_dim)

        self.f_mu.weight.data.uniform_(-1e-3, 1e-3)
        self.f_mu.bias.data.uniform_(-1e-3, 1e-3)
        self.f_logvar.weight.data.uniform_(-1e-3, 1e-3)
        self.f_logvar.bias.data.uniform_(-1e-3, 1e-3)
        self.f_lambda.weight.data.uniform_(-1e-3, 1e-3)
        self.f_lambda.bias.data.uniform_(-1e-3, 1e-3)
        self.f_beta.weight.data.uniform_(-1e-3, 1e-3)
        self.f_beta.bias.data.uniform_(-1e-3, 1e-3)


        # in_layers = 0
        # out_layers = 0
        # self.fc_in = nn.ModuleList([nn.Linear(self.latent_sizes[0], self.latent_sizes[0]) for i in range(in_layers)])
        # self.bn_in = nn.ModuleList([nn.BatchNorm1d(self.latent_sizes[0]) for i in range(in_layers)])
        # self.fc_out = nn.ModuleList([nn.Linear(self.representation_size, self.representation_size) for i in range(out_layers)])
        # self.bn_out = nn.ModuleList([nn.BatchNorm1d(self.representation_size) for i in range(out_layers)])
        # self.dropout = nn.Dropout(0.5)
        # for f in self.fc_in:
        #     f.weight.data.uniform_(-1e-3, 1e-3)
        #     f.bias.data.uniform_(-1e-3, 1e-3)
        # for f in self.fc_out:
        #     f.weight.data.uniform_(-1e-3, 1e-3)
        #     f.bias.data.uniform_(-1e-3, 1e-3)
        # for f in self.bn_in:
        #     f.weight.data.uniform_(-1e-3, 1e-3)
        #     f.bias.data.uniform_(-1e-3, 1e-3)
        # for f in self.bn_out:
        #     f.weight.data.uniform_(-1e-3, 1e-3)
        #     f.bias.data.uniform_(-1e-3, 1e-3)

    def compute_loss(self, obs, cond):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        z = self._pre_vq_conv(z)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        z_s, kle = self.reparameterize(z, z_cond)

        vq_loss, quantized, perplexity, _ = self._vq_vae(z_s.detach()) 
        cond_vq_loss, cond_quantized, cond_perplexity, _ = self.cond_vq_vae(z_cond.detach())
        
        cat_quantized = torch.cat([z_s, z_cond], dim=1)
        
        # for i in range(len(self.fc_out)):
        #     cat_quantized = self.fc_out[i](self.dropout(self.bn_out[i]((F.relu(cat_quantized)))))
        #cat_quantized = cat_quantized.reshape(-1, self.embedding_dim * 2, self.root_len, self.root_len)

        cond_recon = self.cond_decoder(z_cond)
        x_recon = self._decoder(cat_quantized)
        vq_losses = [vq_loss, cond_vq_loss]
        perplexities = [perplexity, cond_perplexity]
        recons = [x_recon, cond_recon]
        errors = [F.mse_loss(x_recon, obs, reduction='sum'), F.mse_loss(cond_recon, cond, reduction='sum')]
        return vq_losses, perplexities, recons, errors, kle


    def latent_to_square(self, latents):
        latents = latents.reshape(-1, 2, self.root_len, self.root_len)
        return latents[:, 0], latents[:, 1]

    def encode(self, obs, cond, cont=True):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)

        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)

        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)
        z_s, kle = self.reparameterize(z, z_cond)

        vq_loss, quantized, perplexity, _ = self._vq_vae(z_s.detach())
        cond_vq_loss, cond_quantized, cond_perplexity, _ = self.cond_vq_vae(z_cond.detach())

        cat_quantized = torch.cat([z_s, z_cond], dim=1)

        return cat_quantized.reshape(obs.shape[0], -1)

    def reparameterize(self, latent, z_cond):
        z_cond = z_cond.reshape(-1, self.latent_sizes[1])
        latent = latent.reshape(-1, self.embedding_dim, self.discrete_size)
        lamb = torch.sigmoid(self.f_lambda(z_cond)).reshape(-1, self.embedding_dim, 1)
        beta =  self.f_beta(z_cond).reshape(-1, self.embedding_dim, 1)
        latent = (latent * lamb + beta).reshape(-1, self.latent_sizes[0])

        mu = self.f_mu(latent)
        logvar = self.log_min_variance + self.f_logvar(latent)
        
        if self.training: z = self.rsample(mu, logvar)
        else: z = mu
        
        z = z.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
        kle = self.kl_divergence(mu, logvar)
        return z, kle

    def kl_divergence(self, mu, logvar):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def rsample(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        #stds = stds.repeat(mu.shape[0], 1).reshape(*mu.size())
        epsilon = ptu.randn(*mu.size())
        latents = epsilon * stds + mu
        return latents

    def conditioned_discrete_to_cont(self, e_indices):
        z_ind, cond_ind = self.latent_to_square(e_indices)
        z = self.discrete_to_cont(z_ind, self._vq_vae._embedding.weight)
        z_cond = self.discrete_to_cont(cond_ind, self.cond_vq_vae._embedding.weight)
        cat_quantized = torch.cat([z, z_cond], dim=1)
        return cat_quantized


    def discrete_to_cont(self, e_indices, e_weights):
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)
        
        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)
        
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)
        
        z_q = torch.matmul(min_encodings, e_weights).view(input_shape) 
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn

    def decode(self, latents, cont=True):
        if cont:
            # for i in range(len(self.fc_out)):
            #     latents = self.fc_out[i](self.dropout(self.bn_out[i](F.relu(latents))))
            z = latents.reshape(-1, self.embedding_dim * 2, self.root_len, self.root_len)
            #z = latents.reshape(-1, 2 * self.embedding_dim, self.root_len, self.root_len)
        else:
            z = self.conditioned_discrete_to_cont(latents)
        return self._decoder(z)

    def sample_prior(self, batch_size, cond, cont=True):
        if cond.shape[0] == 1:
            cond = cond.repeat(batch_size, 1)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        _, cond_quantized, _, cond_encodings = self.cond_vq_vae(z_cond)

        z = ptu.randn(batch_size, self.embedding_dim, self.root_len, self.root_len)
        _, quantized, _, encodings = self._vq_vae(z)

        cat_quantized = torch.cat([z, z_cond], dim=1)

        return cat_quantized.reshape(-1, self.representation_size)


        if cont:
            z, z_c = z, cond_quantized
            #z, z_c = quantized, cond_quantized
        else:
            z, z_c = encodings, cond_encodings

        z = z.reshape(batch_size, -1)
        z_c = z_c.reshape(batch_size, -1)
        z_cond = torch.cat([z, z_c], dim=1)
        return z_cond


class CVQVAEFiLM(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        min_variance=1e-3,
        imsize=48,
        decay=0):
        super(CVQVAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings
        self.log_min_variance = float(np.log(min_variance))

        #Calculate latent sizes
        if imsize == 48: self.root_len = 12
        elif imsize == 84: self.root_len = 21
        else: raise ValueError(imsize)

        self.discrete_size = self.root_len * self.root_len
        self.latent_sizes = [self.discrete_size * self.embedding_dim, self.discrete_size * self.embedding_dim]
        self.representation_size = sum(self.latent_sizes)
        #Calculate latent sizes

        self._encoder = CondEncoder(input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.f_mu = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self.f_logvar = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self.cond_pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
            self.cond_vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)

            self.cond_vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                              commitment_cost)

        self._decoder = CondDecoder(self.embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)


    def compute_loss(self, obs, cond):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        #inputs = torch.cat([obs, cond], dim=1)
        z_delta, z_cond = self._encoder(obs, cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        z_s, kle = self.reparameterize(z_delta)

        vq_loss, quantized, perplexity, _ = self._vq_vae(z_s.detach()) 
        cond_vq_loss, cond_quantized, cond_perplexity, _ = self.cond_vq_vae(z_cond.detach())
        
        #cat_quantized = torch.cat([z_s, z_cond], dim=1)

        x_recon, cond_recon = self._decoder(z_s, z_cond, return_both=True)
        vq_losses = [vq_loss, cond_vq_loss]
        perplexities = [perplexity, cond_perplexity]
        recons = [x_recon, cond_recon]
        errors = [F.mse_loss(x_recon, obs, reduction='sum'), F.mse_loss(cond_recon, cond, reduction='sum')]
        return vq_losses, perplexities, recons, errors, kle


    def latent_to_square(self, latents):
        latents = latents.reshape(-1, 2, self.root_len, self.root_len)
        return latents[:, 0], latents[:, 1]

    def encode(self, obs, cond, cont=True):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        #inputs = torch.cat([obs, cond], dim=1)

        z_delta, z_cond = self._encoder(obs, cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        z_s, kle = self.reparameterize(z_delta)

        vq_loss, quantized, perplexity, _ = self._vq_vae(z_s.detach())
        cond_vq_loss, cond_quantized, cond_perplexity, _ = self.cond_vq_vae(z_cond.detach())

        cat_quantized = torch.cat([z_s, z_cond], dim=1)

        return cat_quantized.reshape(-1, self.representation_size)

    def reparameterize(self, latent):
        mu = self.f_mu(latent).reshape(-1, self.latent_sizes[0])
        logvar = (self.log_min_variance + self.f_logvar(latent)).reshape(-1, self.latent_sizes[0])
        
        if self.training: z_s = self.rsample(mu, logvar)
        else: z_s = mu
        
        z_s = z_s.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
        kle = self.kl_divergence(mu, logvar)
        return z_s, kle

    def kl_divergence(self, mu, logvar):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def rsample(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = ptu.randn(*mu.size())
        latents = epsilon * stds + mu
        return latents

    def conditioned_discrete_to_cont(self, e_indices):
        z_ind, cond_ind = self.latent_to_square(e_indices)
        z = self.discrete_to_cont(z_ind, self._vq_vae._embedding.weight)
        z_cond = self.discrete_to_cont(cond_ind, self.cond_vq_vae._embedding.weight)
        cat_quantized = torch.cat([z, z_cond], dim=1)
        return cat_quantized


    def discrete_to_cont(self, e_indices, e_weights):
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)
        
        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)
        
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)
        
        z_q = torch.matmul(min_encodings, e_weights).view(input_shape) 
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn

    def decode(self, latents, cont=True):
        if cont:
            # for i in range(len(self.fc_out)):
            #     latents = self.fc_out[i](self.dropout(self.bn_out[i](F.relu(latents))))
            z = latents.reshape(-1, self.embedding_dim * 2, self.root_len, self.root_len)
            z_delta, z_cond = z[:, :self.embedding_dim], z[:, self.embedding_dim:]
            return self._decoder(z_delta, z_cond)
            #z = latents.reshape(-1, 2 * self.embedding_dim, self.root_len, self.root_len)
        else:
            z = self.conditioned_discrete_to_cont(latents)
        return self._decoder(z)

    def sample_prior(self, batch_size, cond, cont=True):
        if cond.shape[0] == 1:
            cond = cond.repeat(batch_size, 1)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)
        z_cond = self._encoder.encode_cond(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        _, cond_quantized, _, cond_encodings = self.cond_vq_vae(z_cond)

        z = ptu.randn(batch_size, self.embedding_dim, self.root_len, self.root_len)
        _, quantized, _, encodings = self._vq_vae(z)

        cat_quantized = torch.cat([z, z_cond], dim=1)

        return cat_quantized.reshape(-1, self.representation_size)


        if cont:
            z, z_c = z, cond_quantized
            #z, z_c = quantized, cond_quantized
        else:
            z, z_c = encodings, cond_encodings

        z = z.reshape(batch_size, -1)
        z_c = z_c.reshape(batch_size, -1)
        z_cond = torch.cat([z, z_c], dim=1)
        return z_cond


class CVQVAEQuantize(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        min_variance=1e-3,
        imsize=48,
        decay=0.0):
        super(CVQVAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings
        self.log_min_variance = float(np.log(min_variance))

        #Calculate latent sizes
        if imsize == 48: self.root_len = 12
        elif imsize == 84: self.root_len = 21
        else: raise ValueError(imsize)

        self.discrete_size = self.root_len * self.root_len
        self.latent_sizes = [self.discrete_size * self.embedding_dim, self.discrete_size * self.embedding_dim]
        self.representation_size = sum(self.latent_sizes)
        #Calculate latent sizes

        self._encoder = Encoder(input_channels * 2, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.cond_encoder = Encoder(input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.f_mu = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self.f_logvar = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self.cond_pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
            self.cond_vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)

            self.cond_vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                              commitment_cost)

        self._decoder = Decoder(self.embedding_dim * 2,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self.cond_decoder = Decoder(self.embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)


    def compute_loss(self, obs, cond):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        z_delta = self._encoder(inputs)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        z_s, kle, vq_loss, z_quant, perplexity, _ = self.reparameterize(z_delta, return_loss=True)
        cond_vq_loss, cond_quant, cond_perplexity, _ = self.cond_vq_vae(z_cond)
        
        cat_quantized = torch.cat([z_s, cond_quant], dim=1)
        x_recon = self._decoder(cat_quantized)
        cond_recon = self.cond_decoder(cond_quant)
        vq_losses = [vq_loss, cond_vq_loss]
        perplexities = [perplexity, cond_perplexity]
        recons = [x_recon, cond_recon]
        errors = [F.mse_loss(x_recon, obs, reduction='sum'), F.mse_loss(cond_recon, cond, reduction='sum')]
        return vq_losses, perplexities, recons, errors, kle


    def latent_to_square(self, latents):
        latents = latents.reshape(-1, 2, self.root_len, self.root_len)
        return latents[:, 0], latents[:, 1]

    def encode(self, obs, cond, cont=True):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)

        z_delta = self._encoder(inputs)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        z_s = self.reparameterize(z_delta)
        cond_vq_loss, cond_quant, cond_perplexity, _ = self.cond_vq_vae(z_cond)

        cat_quantized = torch.cat([z_s, cond_quant], dim=1)

        return cat_quantized.reshape(-1, self.representation_size)

    def reparameterize(self, latent, return_loss=False):
        mu = self.f_mu(latent)
        logvar = self.log_min_variance + self.f_logvar(latent)

        vq_loss, z_quant, perplexity, e_indices = self._vq_vae(mu)
        z_quant = z_quant.reshape(-1, self.latent_sizes[0])
        mu = mu.reshape(-1, self.latent_sizes[0])
        logvar = logvar.reshape(-1, self.latent_sizes[0])
        
        # if self.training: z_s = self.rsample(mu, logvar)
        # else: z_s = mu
        if self.training: z_s = self.rsample(z_quant, logvar)
        else: z_s = z_quant
        
        z_s = z_s.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
        
        if return_loss:
            kle = self.kl_divergence(mu, logvar)
            return z_s, kle, vq_loss, z_quant, perplexity, e_indices

        return z_s

    def kl_divergence(self, mu, logvar):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def rsample(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = ptu.randn(*mu.size())
        latents = epsilon * stds + mu
        return latents

    def conditioned_discrete_to_cont(self, e_indices):
        z_ind, cond_ind = self.latent_to_square(e_indices)
        z = self.discrete_to_cont(z_ind, self._vq_vae._embedding.weight)
        z_cond = self.discrete_to_cont(cond_ind, self.cond_vq_vae._embedding.weight)
        cat_quantized = torch.cat([z, z_cond], dim=1)
        return cat_quantized


    def discrete_to_cont(self, e_indices, e_weights):
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)
        
        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)
        
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)
        
        z_q = torch.matmul(min_encodings, e_weights).view(input_shape) 
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn

    def decode(self, latents, cont=True):
        if cont:
            # for i in range(len(self.fc_out)):
            #     latents = self.fc_out[i](self.dropout(self.bn_out[i](F.relu(latents))))
            z = latents.reshape(-1, self.embedding_dim * 2, self.root_len, self.root_len)
        else:
            z = self.conditioned_discrete_to_cont(latents)
        return self._decoder(z)

    def sample_prior(self, batch_size, cond, cont=True):
        if cond.shape[0] == 1:
            cond = cond.repeat(batch_size, 1)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        z = ptu.randn(batch_size, self.embedding_dim, self.root_len, self.root_len)
        
        _, cond_quant, _, cond_encodings = self.cond_vq_vae(z_cond)
        _, z_quant, _, encodings = self._vq_vae(z)

        if cont:
            z, z_c = z_quant, cond_quant
        else:
            z, z_c = encodings, cond_encodings

        z = z.reshape(batch_size, -1)
        z_c = z_c.reshape(batch_size, -1)
        z_cond = torch.cat([z, z_c], dim=1)
        return z_cond


class VQ_VAE_STANDARD(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        imsize=48,
        decay=0.0):
        super(VQ_VAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings
        self._encoder = Encoder(input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(self.embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        
        #Calculate latent sizes
        if imsize == 48: self.root_len = 12
        elif imsize == 84: self.root_len = 21
        else: raise ValueError(imsize)

        self.discrete_size = self.root_len * self.root_len
        self.representation_size = self.discrete_size * self.embedding_dim
        #Calculate latent sizes

    def compute_loss(self, inputs):
        inputs = inputs.view(-1,
                            self.input_channels,
                            self.imsize,
                            self.imsize)
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)
        vq_loss, quantized, perplexity, _ = self._vq_vae(z)

        x_recon = self._decoder(quantized)
        recon_error = F.mse_loss(x_recon, inputs)
        return vq_loss, quantized, x_recon, perplexity, recon_error


    def latent_to_square(self, latents):
        #root_len = int(latents.shape[1] ** 0.5)
        return latents.reshape(-1, self.root_len, self.root_len)

    def encode(self, inputs, cont=True):
        inputs = inputs.view(-1,
                            self.input_channels,
                            self.imsize,
                            self.imsize)
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)
        _, quantized, _, encodings = self._vq_vae(z)

        if cont:
            return quantized.reshape(inputs.shape[0], -1)

        return encodings.reshape(inputs.shape[0], -1)


    def encode_one_np(self, inputs, cont=True):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))[0]


    def encode_np(self, inputs, cont=True):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))

    def decode_one_np(self, inputs, cont=True):
        return np.clip(ptu.get_numpy(self.decode(ptu.from_numpy(inputs).reshape(1, -1), cont=cont))[0], 0, 1)


    def decode_np(self, inputs, cont=True):
        return np.clip(ptu.get_numpy(self.decode(ptu.from_numpy(inputs), cont=cont)), 0, 1)


    def discrete_to_cont(self, e_indices):
        e_indices = self.latent_to_square(e_indices)
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)

        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)

        e_weights = self._vq_vae._embedding.weight
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)

        z_q = torch.matmul(min_encodings, e_weights).view(input_shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn


    def decode(self, latents, cont=True):
        z_q = None
        if cont:
            z_q = latents.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
        else:
            z_q = self.discrete_to_cont(latents)

        return self._decoder(z_q)


class VAE(nn.Module):
    def __init__(
        self,
        representation_size,
        embedding_dim=3,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        min_variance=1e-3,
        imsize=48,
        ):
        super(VAE, self).__init__()
        self.log_min_variance = float(np.log(min_variance))
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        
        self._encoder = Encoder(input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_rep_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)

        self._decoder = Decoder(self.embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        
        #Calculate latent sizes
        if imsize == 48:
            self.root_conv_size = 12
        elif imsize == 84:
            self.root_conv_size = 21
        else:
            raise ValueError(imsize)

        self.conv_size = self.root_conv_size * self.root_conv_size * self.embedding_dim
        self.representation_size = representation_size
        #Calculate latent sizes

        assert representation_size < self.conv_size  # This is a bad idea (wrong bottleneck)

        self.f_mu = nn.Linear(self.conv_size, self.representation_size)
        self.f_logvar = nn.Linear(self.conv_size, self.representation_size)
        self.f_dec = nn.Linear(self.representation_size, self.conv_size)

        #self.dropout = nn.Dropout(0.5)
        #self.bn = nn.BatchNorm1d(self.conv_size)

        self.f_mu.weight.data.uniform_(-1e-3, 1e-3)
        self.f_mu.bias.data.uniform_(-1e-3, 1e-3)
        self.f_logvar.weight.data.uniform_(-1e-3, 1e-3)
        self.f_logvar.bias.data.uniform_(-1e-3, 1e-3)
        self.f_dec.weight.data.uniform_(-1e-3, 1e-3)
        self.f_dec.bias.data.uniform_(-1e-3, 1e-3)

    def compute_loss(self, obs):
        obs = obs.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)

        z_s, kle = self.encode_image(obs)

        recon = self.decode(z_s)
        recon_error = F.mse_loss(recon, obs, reduction='sum')

        return recon, recon_error, kle


    def encode_image(self, obs):
        obs = obs.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)

        z_conv = self._encoder(obs)
        z_conv = self._pre_rep_conv(z_conv)

        return self.compress(z_conv)

    def compress(self, z_conv):
        z_conv = z_conv.view(-1, self.conv_size)

        mu = self.f_mu(z_conv)
        logvar = self.log_min_variance + self.f_logvar(z_conv)
        if self.training:
          z_s = self.rsample(mu, logvar)
        else:
          z_s = mu

        kle = self.kl_divergence(mu, logvar)
        
        return z_s, kle

    def decompress(self, z_s):
        z_conv = self.f_dec(z_s)
        z_conv = z_conv.view(-1, self.embedding_dim, self.root_conv_size, self.root_conv_size)   
        return z_conv

    def rsample(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = ptu.randn(*mu.size())
        latents = epsilon * stds + mu
        return latents

    def kl_divergence(self, mu, logvar):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def encode(self, inputs):
        z_s, _ = self.encode_image(inputs)
        return z_s

    def sample_prior(self, batch_size):
        z_s = ptu.randn(batch_size, self.representation_size)
        return z_s

    def decode(self, latents):
        z_conv = self.decompress(latents)
        return self._decoder(z_conv)

# class CVAE(nn.Module):
#     def __init__(
#         self,
#         embedding_dim,
#         root_len=21,
#         input_channels=3,
#         num_hiddens=128,
#         num_residual_layers=3,
#         num_residual_hiddens=64,
#         num_embeddings=512,
#         commitment_cost=0.25,
#         decoder_output_activation=None, #IGNORED FOR NOW
#         architecture=None, #IGNORED FOR NOW
#         imsize=48,
#         decay=0.0):
#         super(CVAE, self).__init__()
#         self.imsize = imsize
#         self.embedding_dim = embedding_dim
#         self.pixel_cnn = None
#         self.input_channels = input_channels
#         self.imlength = imsize * imsize * input_channels
#         self.num_embeddings = num_embeddings
        
#         self._encoder = Encoder(2 * input_channels, num_hiddens,
#                                 num_residual_layers,
#                                 num_residual_hiddens)
#         self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
#                                       out_channels=self.embedding_dim,
#                                       kernel_size=1,
#                                       stride=1)


#         self._cond_encoder = Encoder(input_channels, num_hiddens,
#                                 num_residual_layers,
#                                 num_residual_hiddens)
        
#         self._cond_pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
#                                       out_channels=self.embedding_dim,
#                                       kernel_size=1,
#                                       stride=1)

#         #FINISH THIS!!!

#         self._decoder = Decoder(2 * self.embedding_dim,
#                                 num_hiddens,
#                                 num_residual_layers,
#                                 num_residual_hiddens)

#         # self._cond_decoder = Decoder(self.embedding_dim,
#         #                         num_hiddens,
#         #                         num_residual_layers,
#         #                         num_residual_hiddens)
        
#         #Calculate latent sizes
#         if imsize == 48:
#             self.root_conv_size = 12
#         elif imsize == 84:
#             self.root_conv_size = 21
#         else:
#             raise ValueError(imsize)

#         #assert root_len < self.root_conv_size  # This is a bad idea (wrong bottleneck)
#         self.root_len = root_len
#         self.discrete_size = root_len * root_len
#         self.conv_size = self.root_conv_size * self.root_conv_size * self.embedding_dim
#         self.representation_size = self.discrete_size * self.embedding_dim
#         #Calculate latent sizes

#         self.f_mu = nn.Linear(self.conv_size, self.representation_size)
#         self.f_logvar = nn.Linear(self.conv_size, self.representation_size)
#         #self.f_c = nn.Linear(self.conv_size, self.representation_size)
#         self.f_dec = nn.Linear(self.representation_size, self.conv_size)

#         self.bn_c = nn.BatchNorm1d(self.conv_size)


#     def compute_loss(self, x_delta, x_cond):
#         x_delta = x_delta.view(-1,
#                             self.input_channels,
#                             self.imsize,
#                             self.imsize)

#         z_cat, kle = self.encode_images(x_delta, x_cond)

#         x_recon = self.decode(z_cat)
#         recon_error = F.mse_loss(x_recon, x_delta, reduction='sum')

#         return x_recon, recon_error, kle


#     def encode_images(self, x_delta, x_cond):
#         x_delta = x_delta.view(-1,
#                             self.input_channels,
#                             self.imsize,
#                             self.imsize)
#         x_cond = x_cond.view(-1,
#                             self.input_channels,
#                             self.imsize,
#                             self.imsize)

#         x_delta = torch.cat([x_delta, x_cond], dim=1)

#         z_delta = self._encoder(x_delta)
#         z_delta = self._pre_vq_conv(z_delta)

#         z_cond = self._cond_encoder(x_cond)
#         z_cond = self._cond_pre_vq_conv(z_cond)

#         return self.compress(z_delta, z_cond)

#     def compress(self, z_delta, z_cond):
#         z_delta = z_delta.view(-1, self.conv_size)
#         mu = self.f_mu(z_delta)
#         logvar = self.f_logvar(z_delta)

#         z_cond = z_cond.view(-1, self.conv_size)
#         #z_cond = self.bn_c(z_cond)

#         if self.training: z_s = self.rsample(mu, logvar)
#         else: z_s = mu

#         z_cat = torch.cat([z_s, z_cond], dim=1)
#         kle = self.kl_divergence(mu, logvar)
        
#         return z_cat, kle

#     def decompress(self, latents):
#         z_delta = self.f_dec(latents[:, :self.representation_size])
#         z_cond = latents[:, self.representation_size:]

#         z_delta = z_delta.view(-1, self.embedding_dim, self.root_conv_size, self.root_conv_size)
#         z_cond = z_cond.view(-1, self.embedding_dim, self.root_conv_size, self.root_conv_size)

#         z_cat = torch.cat([z_delta, z_cond], dim=1)    
#         return z_cat

#     def reparameterize(self, mu, logvar):
#         mu = self.f_mu(latent).reshape(-1, self.latent_sizes[0])
#         logvar = (self.log_min_variance + self.f_logvar(latent)).reshape(-1, self.latent_sizes[0])
        
#         if self.training: z_s = self.rsample(mu, logvar)
#         else: z_s = mu
        
#         z_s = z_s.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
#         kle = self.kl_divergence(mu, logvar)
#         return z_s, kle

#     def rsample(self, mu, logvar):
#         stds = (0.5 * logvar).exp()
#         epsilon = ptu.randn(*mu.size())
#         latents = epsilon * stds + mu
#         return latents

#     def kl_divergence(self, mu, logvar):
#         return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

#     def encode(self, inputs, cont=True):
#         z_cat, _ = self.encode_images(inputs)
#         return z_cat

#     def sample_prior(self, batch_size, cond):
#         if cond.shape[0] == 1:
#             cond = cond.repeat(batch_size, 1)

#         cond = cond.view(batch_size,
#                         self.input_channels,
#                         self.imsize,
#                         self.imsize)

#         z_cond = self._cond_encoder(cond)
#         z_cond = self._cond_pre_vq_conv(z_cond)
#         z_cond = z_cond.view(-1, self.conv_size)
#         #z_cond = self.bn_c(z_cond)

#         z_delta = ptu.randn(batch_size, self.representation_size)
#         z_cat = torch.cat([z_delta, z_cond], dim=1)

#         return z_cat

#     def decode(self, latents):
#         z_conv = self.decompress(latents)
#         return self._decoder(z_conv)

#     def encode_one_np(self, inputs, cont=True):
#         return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))[0]

#     def encode_np(self, inputs, cont=True):
#         return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))

#     def decode_one_np(self, inputs, cont=True):
#         return np.clip(ptu.get_numpy(self.decode(ptu.from_numpy(inputs).reshape(1, -1), cont=cont))[0], 0, 1)

#     def decode_np(self, inputs, cont=True):
#         return np.clip(ptu.get_numpy(self.decode(ptu.from_numpy(inputs), cont=cont)), 0, 1)


class CVAE1(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_channels=3,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decoder_output_activation=None, #IGNORED FOR NOW
        architecture=None, #IGNORED FOR NOW
        min_variance=1e-3,
        imsize=48,
        decay=0):
        super(CVAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings
        self.log_min_variance = float(np.log(min_variance))

        #Calculate latent sizes
        if imsize == 48: self.root_len = 12
        elif imsize == 84: self.root_len = 21
        else: raise ValueError(imsize)

        self.discrete_size = self.root_len * self.root_len
        self.latent_sizes = [self.discrete_size * self.embedding_dim, self.discrete_size * self.embedding_dim]
        self.representation_size = sum(self.latent_sizes)
        #Calculate latent sizes

        self._encoder = Encoder(input_channels * 2, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.cond_encoder = Encoder(input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.f_mu = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self.f_logvar = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self.cond_pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
            self.cond_vq_vae = VectorQuantizerEMA(num_embeddings, self.embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                           commitment_cost)

            self.cond_vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                                              commitment_cost)

        self._decoder = Decoder(self.embedding_dim * 2,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self.cond_decoder = Decoder(self.embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)


    def compute_loss(self, obs, cond):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)
        z_delta = self._encoder(inputs)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        z_s, kle = self.reparameterize(z_delta)

        vq_loss, quantized, perplexity, _ = self._vq_vae(z_s.detach()) 
        cond_vq_loss, cond_quantized, cond_perplexity, _ = self.cond_vq_vae(z_cond.detach())
        
        cat_quantized = torch.cat([z_s, z_cond], dim=1)
        x_recon = self._decoder(cat_quantized)
        cond_recon = self.cond_decoder(z_cond)
        vq_losses = [vq_loss, cond_vq_loss]
        perplexities = [perplexity, cond_perplexity]
        recons = [x_recon, cond_recon]
        #errors = [F.mse_loss(x_recon, obs), F.mse_loss(cond_recon, cond)]
        errors = [F.mse_loss(x_recon, obs, reduction='sum'), F.mse_loss(cond_recon, cond, reduction='sum')]
        return vq_losses, perplexities, recons, errors, kle


    def latent_to_square(self, latents):
        latents = latents.reshape(-1, 2, self.root_len, self.root_len)
        return latents[:, 0], latents[:, 1]

    def encode(self, obs, cond, cont=True):
        obs = obs.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        inputs = torch.cat([obs, cond], dim=1)

        z_delta = self._encoder(inputs)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        z_s, kle = self.reparameterize(z_delta)

        vq_loss, quantized, perplexity, _ = self._vq_vae(z_s.detach())
        cond_vq_loss, cond_quantized, cond_perplexity, _ = self.cond_vq_vae(z_cond.detach())

        cat_quantized = torch.cat([z_s, z_cond], dim=1)

        return cat_quantized.reshape(-1, self.representation_size)

    def reparameterize(self, latent):
        mu = self.f_mu(latent).reshape(-1, self.latent_sizes[0])
        logvar = (self.log_min_variance + self.f_logvar(latent)).reshape(-1, self.latent_sizes[0])
        
        if self.training: z_s = self.rsample(mu, logvar)
        else: z_s = mu
        
        z_s = z_s.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
        kle = self.kl_divergence(mu, logvar)
        return z_s, kle

    def kl_divergence(self, mu, logvar):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def rsample(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = ptu.randn(*mu.size())
        latents = epsilon * stds + mu
        return latents

    def conditioned_discrete_to_cont(self, e_indices):
        z_ind, cond_ind = self.latent_to_square(e_indices)
        z = self.discrete_to_cont(z_ind, self._vq_vae._embedding.weight)
        z_cond = self.discrete_to_cont(cond_ind, self.cond_vq_vae._embedding.weight)
        cat_quantized = torch.cat([z, z_cond], dim=1)
        return cat_quantized


    def discrete_to_cont(self, e_indices, e_weights):
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)
        
        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)
        
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)
        
        z_q = torch.matmul(min_encodings, e_weights).view(input_shape) 
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn

    def decode(self, latents, cont=True):
        if cont:
            # for i in range(len(self.fc_out)):
            #     latents = self.fc_out[i](self.dropout(self.bn_out[i](F.relu(latents))))
            z = latents.reshape(-1, self.embedding_dim * 2, self.root_len, self.root_len)
        else:
            z = self.conditioned_discrete_to_cont(latents)
        return self._decoder(z)

    def sample_prior(self, batch_size, cond, cont=True):
        if cond.shape[0] == 1:
            cond = cond.repeat(batch_size, 1)

        cond = cond.view(-1,
                        self.input_channels,
                        self.imsize,
                        self.imsize)
        z_cond = self.cond_encoder(cond)
        z_cond = self.cond_pre_vq_conv(z_cond)
        _, cond_quantized, _, cond_encodings = self.cond_vq_vae(z_cond)

        z = ptu.randn(batch_size, self.embedding_dim, self.root_len, self.root_len)
        _, quantized, _, encodings = self._vq_vae(z)

        cat_quantized = torch.cat([z, z_cond], dim=1)

        return cat_quantized.reshape(-1, self.representation_size)


        if cont:
            z, z_c = z, cond_quantized
            #z, z_c = quantized, cond_quantized
        else:
            z, z_c = encodings, cond_encodings

        z = z.reshape(batch_size, -1)
        z_c = z_c.reshape(batch_size, -1)
        z_cond = torch.cat([z, z_c], dim=1)
        return z_cond


    def spatial_encoder(self, latent, z_cond):
        sofmax = nn.Softmax2d()
        latent = sofmax(latent)

        maps_x = torch.sum(latent, 2)
        maps_y = torch.sum(latent, 3)

        weights = ptu.from_numpy(np.arange(maps_x.shape[-1]) / maps_x.shape[-1])
        fp_x = torch.sum(maps_x * weights, 2)
        fp_y = torch.sum(maps_y * weights, 2)
        latent = torch.cat([fp_x, fp_y, z_cond.reshape(-1, self.latent_sizes[1])], 1)

        mu = self.mu(latent)
        logvar = self.log_min_variance + torch.abs(self.logvar(latent))
        return (mu, logvar)
