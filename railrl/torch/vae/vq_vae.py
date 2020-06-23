from __future__ import print_function
import torch
import numpy as np
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from railrl.pythonplusplus import identity
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

    def forward(self, inputs):
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
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 /
                                             self._num_embeddings, 1/self._num_embeddings)
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
        self.representation_size = 0
        self.discrete_size = 0
        self.square_size = 0

    def compute_loss(self, inputs):
        inputs = inputs.view(-1,
                            self.input_channels,
                            self.imsize,
                            self.imsize)
        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)
        vq_loss, quantized, perplexity, _ = self._vq_vae(z)

        if self.representation_size == 0:
            self.representation_size = quantized[0].flatten().shape[0]
            self.discrete_size = self.representation_size // self.embedding_dim
            self.square_size = int(self.discrete_size ** 0.5)

        x_recon = self._decoder(quantized)
        recon_error = F.mse_loss(x_recon, inputs)
        return vq_loss, quantized, x_recon, perplexity, recon_error


    def latent_to_square(self, latents):
        root_len = int(latents.shape[1] ** 0.5)
        return latents.reshape(-1, root_len, root_len)

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

    # def encode(self, inputs):
    #     inputs = inputs.view(-1,
    #                         self.input_channels,
    #                         self.imsize,
    #                         self.imsize)
    #     z = self._encoder(inputs)
    #     z = self._pre_vq_conv(z)
    #     _, quantized, _, _ = self._vq_vae(z)
    #     return quantized

    def decode(self, latents, cont=True):
        z_q = None
        if cont:
            #squared_len = int(self.discrete_size ** 0.5)
            squared_len = int((latents.shape[1] / self.embedding_dim) ** 0.5)
            z_q = latents.reshape(-1, self.embedding_dim, squared_len, squared_len)
        else:
            z_q = self.discrete_to_cont(latents)

        return self._decoder(z_q)

    # def generate_samples(self, e_indices):
    #     input_shape = e_indices.shape + (self.representation_size,)
    #     e_indices = e_indices.reshape(-1).unsqueeze(1)#, input_shape[1]*input_shape[2])

    #     min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings, device=e_indices.device)
    #     min_encodings.scatter_(1, e_indices, 1)

    #     e_weights = self._vq_vae._embedding.weight
    #     quantized = torch.matmul(
    #         min_encodings, e_weights).view(input_shape)

    #     z_q = torch.matmul(min_encodings, e_weights).view(input_shape)
    #     z_q = z_q.permute(0, 3, 1, 2).contiguous()

    #     x_recon = self._decoder(z_q)
    #     return x_recon

    def get_distance(self, s_indices, g_indices):
        assert s_indices.shape == g_indices.shape
        batch_size = s_indices.shape[0]
        s_q = self.discrete_to_cont(s_indices).reshape(batch_size, -1)
        g_q = self.discrete_to_cont(g_indices).reshape(batch_size, -1)
        return ptu.get_numpy(torch.norm(s_q - g_q, dim=1))

    def sample_prior(self, batch_size, cont=True):
        root_len = int(self.representation_size**0.5)
        e_indices = self.pixel_cnn.generate(shape=(root_len, root_len), batch_size=batch_size)
        e_indices = e_indices.reshape(batch_size, -1)
        if cont:
            return self.discrete_to_cont(e_indices)
        return e_indices

    # def logprob(self, e_indices, cont=True):
    #     batch_size = images.shape[0]
    #     # e_indices = self.encode(images, cont=False)
    #     cond = ptu.from_numpy(np.ones((images.shape[0], 1)))
    #     logits = self.pixel_cnn(e_indices, cond)
    #     logits = logits.permute(0, 2, 3, 1).contiguous()
    #     criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    #     logprob = - criterion(
    #         logits.view(-1, self.num_embeddings),
    #         e_indices.contiguous().view(-1))

    #     logprob = logprob.reshape(batch_size, -1).mean(dim=1)



    #     # logprob = - criterion(
    #     #     logits.view(batch_size, -1, self.num_embeddings),
    #     #     e_indices.contiguous().view(batch_size, -1))
    #     return logprob

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


class CVQVAE(nn.Module):
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

        self._decoder = Decoder(self.embedding_dim * 2,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.cond_decoder = Decoder(self.embedding_dim,
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
        cat_quantized = torch.cat([quantized, cond_quantized], dim=1)

        if self.representation_size == 0:
            z_size = quantized[0].flatten().shape[0]
            z_cond_size = cond_quantized[0].flatten().shape[0]
            self.latent_sizes = [z_size, z_cond_size]
            self.representation_size = z_size + z_cond_size
            self.discrete_size = self.representation_size // self.embedding_dim
            self.root_len = int((self.discrete_size // 2) ** 0.5)
        
        cond_recon = self.cond_decoder(cond_quantized)
        x_recon = self._decoder(cat_quantized)
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
            z = self.conditioned_discrete_to_cont(latents)
        return self._decoder(z)


    # def sample_prior(self, batch_size, cont=True):
    #     size = self.latent_sizes[0]**0.5
    #     e_indices = self.pixel_cnn.generate(shape=(size, size), batch_size=batch_size)
    #     e_indices = e_indices.reshape(batch_size, -1)
    #     if cont:
    #         return self.discrete_to_cont(e_indices)
    #     return e_indices
