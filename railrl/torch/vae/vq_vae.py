from __future__ import print_function
import torch
import numpy as np
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy, MlpPolicy
from railrl.torch.networks import CNN, TwoHeadDCNN, DCNN
from railrl.torch.vae.vae_base import compute_bernoulli_log_prob, compute_gaussian_log_prob, GaussianLatentVAE
from railrl.torch.vae.conv_vae import ConvVAE
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         try:
#             nn.init.xavier_uniform_(m.weight.data)
#             m.bias.data.fill_(0)
#         except AttributeError:
#             print("Skipping initialization of ", classname)



# class GatedActivation(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         x, y = x.chunk(2, dim=1)
#         return torch.tanh(x) * torch.sigmoid(y)


# class GatedMaskedConv2d(nn.Module):
#     def __init__(self, mask_type, dim, kernel, residual=True, n_classes=1):
#         super().__init__()
#         assert kernel % 2 == 1, print("Kernel size must be odd")
#         self.mask_type = mask_type
#         self.residual = residual

#         self.class_cond_embedding = nn.Embedding(
#             n_classes, 2 * dim
#         )

#         kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
#         padding_shp = (kernel // 2, kernel // 2)
#         self.vert_stack = nn.Conv2d(
#             dim, dim * 2,
#             kernel_shp, 1, padding_shp
#         )

#         self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

#         kernel_shp = (1, kernel // 2 + 1)
#         padding_shp = (0, kernel // 2)
#         self.horiz_stack = nn.Conv2d(
#             dim, dim * 2,
#             kernel_shp, 1, padding_shp
#         )

#         self.horiz_resid = nn.Conv2d(dim, dim, 1)

#         self.gate = GatedActivation()

#     def make_causal(self):
#         self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
#         self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

#     def forward(self, x_v, x_h, h):
#         if self.mask_type == 'A':
#             self.make_causal()

#         h = self.class_cond_embedding(h)
#         h_vert = self.vert_stack(x_v)
#         h_vert = h_vert[:, :, :x_v.size(-1), :]
#         out_v = self.gate(h_vert + h[:, :, None, None])

#         h_horiz = self.horiz_stack(x_h)
#         h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
#         v2h = self.vert_to_horiz(h_vert)

#         out = self.gate(v2h + h_horiz + h[:, :, None, None])
#         if self.residual:
#             out_h = self.horiz_resid(out) + x_h
#         else:
#             out_h = self.horiz_resid(out)

#         return out_v, out_h


# class GatedPixelCNN(nn.Module):
#     def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=1):
#         super().__init__()
#         self.dim = dim

#         # Create embedding layer to embed input
#         self.embedding = nn.Embedding(input_dim, dim)

#         # Building the PixelCNN layer by layer
#         self.layers = nn.ModuleList()

#         # Initial block with Mask-A convolution
#         # Rest with Mask-B convolutions
#         for i in range(n_layers):
#             mask_type = 'A' if i == 0 else 'B'
#             kernel = 7 if i == 0 else 3
#             residual = False if i == 0 else True

#             self.layers.append(
#                 GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
#             )

#         # Add the output layer
#         self.output_conv = nn.Sequential(
#             nn.Conv2d(dim, 512, 1),
#             nn.ReLU(True),
#             nn.Conv2d(512, input_dim, 1)
#         )

#         self.apply(weights_init)

#     def forward(self, x):
#         param = next(self.parameters())
#         shp = x.size() + (-1, )
#         x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
#         x = x.permute(0, 3, 1, 2)  # (B, C, W, H)
#         label = torch.zeros(1, dtype=torch.long, device=param.device)

#         x_v, x_h = (x, x)
#         for i, layer in enumerate(self.layers):
#             x_v, x_h = layer(x_v, x_h, label)

#         return self.output_conv(x_h)

#     def generate(self, shape=(12, 12), batch_size=64):
#         param = next(self.parameters())
#         x = torch.zeros(
#             (batch_size, *shape),
#             dtype=torch.int64, device=param.device
#         )

#         for i in range(shape[0]):
#             for j in range(shape[1]):
#                 logits = self.forward(x) #might need to convert 0 to long
#                 probs = F.softmax(logits[:, :, i, j], -1)
#                 x.data[:, i, j].copy_(
#                     probs.multinomial(1).squeeze().data
#                 )
#         return x



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
        representation_size,
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
        self.representation_size = representation_size
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings
        #self.pixel_cnn = GatedPixelCNN(num_embeddings, 144, 15)
        self._encoder = Encoder(input_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=representation_size,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, representation_size,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, representation_size,
                                           commitment_cost)
        self._decoder = Decoder(representation_size,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

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
        squared_len = int(latents.shape[1] ** 0.5)
        return latents.reshape(-1, squared_len, squared_len)

    def encode(self, inputs, cont=False):
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

    def discrete_to_cont(self, e_indices):
        e_indices = self.latent_to_square(e_indices)
        input_shape = e_indices.shape + (self.representation_size,)
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

    def decode(self, latents, cont=False):
        z_q = None
        if cont:
            squared_len = int((latents.shape[1] / self.representation_size) ** 0.5)
            z_q = latents.reshape(-1, self.representation_size, squared_len, squared_len)
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

    def sample_prior(self, batch_size):
        e_indices = self.pixel_cnn.generate(shape=(12, 12), batch_size=batch_size)
        return e_indices.reshape(batch_size, -1)

