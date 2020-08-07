import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from rlkit.torch.core import PyTorchModule
import torchvision.utils as vutils
from rlkit.torch import pytorch_util as ptu

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)

class BiGAN(PyTorchModule):
    def __init__(self,
        representation_size=12,
        input_channels=3,
        dropout=0.2,
        imsize=48,
        architecture=None, # Not used
        ):
        super().__init__()
        self.representation_size = representation_size
        self.imlength = input_channels * imsize * imsize
        self.input_channels = input_channels
        self.imsize = imsize

        self.netE = Encoder(representation_size, input_channels=input_channels, imsize=imsize, noise=True)
        self.netG = Generator(representation_size, input_channels=input_channels, imsize=imsize)
        self.netD = Discriminator(representation_size, input_channels=input_channels, imsize=imsize, dropout=dropout)
        
        self.netE.apply(weights_init)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

class Generator(nn.Module):
    def __init__(self, representation_size, input_channels=3, imsize=48):
        super(Generator, self).__init__()
        self.representation_size = representation_size
        self.input_channels = input_channels
        self.imsize = imsize

        self.output_bias = nn.Parameter(torch.zeros(self.input_channels, self.imsize, self.imsize), requires_grad=True)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.representation_size, 256, 4, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, input_channels, 1, stride=1, bias=False)
        )

    def forward(self, input):
        input = input.reshape(-1, self.input_channels, self.imsize, self.imsize)
        output = self.main(input)
        output = torch.sigmoid(output + self.output_bias)
        return output



class Encoder(nn.Module):
    def __init__(self, representation_size, input_channels=3, imsize=48, noise=False):
        super(Encoder, self).__init__()
        self.representation_size = representation_size
        self.input_channels = input_channels
        self.imsize = imsize

        self.main1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

        self.main2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )

        self.main3 = nn.Sequential(
            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )

        self.main4 = nn.Sequential(
            nn.Conv2d(512, self.representation_size, 1, stride=1, bias=True)
        )

    def forward(self, input):
        input = input.reshape(-1, self.input_channels, self.imsize, self.imsize)
        batch_size = input.size()[0]
        x1 = self.main1(input)
        x2 = self.main2(x1)
        x3 = self.main3(x2)
        output = self.main4(x3).view(batch_size, self.representation_size, 1, 1)
        return output, x3.view(batch_size, -1), x2.view(batch_size, -1), x1.view(batch_size, -1)



class Discriminator(nn.Module):

    def __init__(self, representation_size, input_channels=3, imsize=48, dropout=0.2):
        super(Discriminator, self).__init__()
        self.representation_size = representation_size
        self.input_channels = input_channels
        self.imsize = imsize
        self.dropout = dropout

        self.infer_x = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(64, 128, 4, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_z = nn.Sequential(
            nn.Conv2d(self.representation_size, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_joint = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.final = nn.Conv2d(1024, 1, 1, stride=1, bias=True)

    def forward(self, x, z):
        output_x = self.infer_x(x)
        output_z = self.infer_z(z)
        output_features = self.infer_joint(torch.cat([output_x, output_z], dim=1))
        output = self.final(output_features)
        output = torch.sigmoid(output)
        return output.squeeze(), output_features.view(x.size()[0], -1)




class ConditionalBiGAN(PyTorchModule):
    def __init__(
            self,
            representation_size=12,
            input_channels=3,
            dropout=0.2,
            imsize=48,
            architecture=None, # Not used
            ):
        super().__init__()
        self.representation_size = representation_size * 2
        self.latent_size = representation_size
        self.imlength = input_channels * imsize * imsize
        self.input_channels = input_channels
        self.imsize = imsize

        self.netE = ConditionalBiGANEncoder(representation_size, input_channels=input_channels, imsize=imsize, noise=True)
        self.netG = ConditionalBiGANGenerator(representation_size, input_channels=input_channels, imsize=imsize)
        self.netD = ConditionalBiGANDiscriminator(representation_size, input_channels=input_channels, imsize=imsize, dropout=dropout)
        
        self.netE.apply(weights_init)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

    def sample_prior(self, batch_size, x_0):
        if x_0.shape[0] == 1:
            x_0 = x_0.repeat(batch_size, 1)

        x_0 = x_0.reshape(-1, self.input_channels, self.imsize, self.imsize)
        z_cond, _, _, _= self.netE.cond_encoder(x_0)
        z_delta = ptu.randn(batch_size, self.latent_size, 1, 1)
        cond_sample = torch.cat([z_delta, z_cond], dim=1)
        return cond_sample

class ConditionalBiGANGenerator(nn.Module):
    def __init__(self, representation_size, input_channels=3, imsize=48):
        super(ConditionalBiGANGenerator, self).__init__()
        self.representation_size = representation_size * 2
        self.latent_size = representation_size
        self.input_channels = input_channels
        self.imsize = imsize

        self.output_bias = nn.Parameter(torch.zeros(self.input_channels, self.imsize, self.imsize), requires_grad=True)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.representation_size, 256, 4, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 1, stride=1, bias=False)
        )

    def forward(self, input):
        output = self.main(input)
        output = torch.sigmoid(output + self.output_bias)
        return output



class ConditionalBiGANEncoder(nn.Module):
    def __init__(self, representation_size, input_channels=3, imsize=48, noise=False):
        super(ConditionalBiGANEncoder, self).__init__()
        self.representation_size = representation_size * 2
        self.latent_size = representation_size
        self.input_channels = input_channels
        self.imsize = imsize

        self.encoder = Encoder(self.latent_size, input_channels=input_channels * 2, imsize=imsize, noise=True)
        self.cond_encoder = Encoder(self.latent_size, input_channels=input_channels, imsize=imsize, noise=True)

    def forward(self, x_delta, x_cond):
        batch_size = x_delta.size()[0]
        x_delta = torch.cat([x_delta, x_cond], dim=1)
        z_delta, _, _, _= self.encoder(x_delta)
        z_cond, _, _, _= self.cond_encoder(x_cond)
        output = torch.cat([z_delta, z_cond], dim=1)
        return output, None, None, None



class ConditionalBiGANDiscriminator(nn.Module):

    def __init__(self, representation_size, input_channels=3, imsize=48, dropout=0.2):
        super(ConditionalBiGANDiscriminator, self).__init__()
        self.representation_size = representation_size * 2
        self.latent_size = representation_size
        self.input_channels = input_channels
        self.imsize = imsize
        self.dropout = dropout

        self.infer_x = nn.Sequential(
            nn.Conv2d(input_channels * 2, 32, 5, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(64, 128, 4, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_z = nn.Sequential(
            nn.Conv2d(self.representation_size, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_joint = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.final = nn.Conv2d(1024, 1, 1, stride=1, bias=True)

    def forward(self, x, x_cond, z):
        obs = torch.cat([x, x_cond], dim=1)
        output_obs = self.infer_x(obs)
        output_z = self.infer_z(z)
        output_features = self.infer_joint(torch.cat([output_obs, output_z], dim=1))
        output = self.final(output_features)
        output = torch.sigmoid(output)
        return output.squeeze(), output_features.view(obs.size()[0], -1)
