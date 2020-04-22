import argparse
from collections import OrderedDict
import os
from os import path as osp
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from railrl.core.loss import LossFunction
from railrl.core import logger
from torchvision.utils import save_image

class BiGANTrainer():

    def __init__(self, model, ngpu, lr, beta, latent_size, generator_threshold):
        self.model = model
        self.netE = self.model.netE
        self.netG = self.model.netG
        self.netD = self. model.netD
        self.device = self.model.device

        self.img_list = []
        self.G_losses = {}
        self.D_losses = {}
        self.iters = 0
        self.criterion = nn.BCELoss()
        
        self.ngpu = ngpu
        self.lr = lr
        self.beta = beta
        self.latent_size = latent_size
        self.generator_threshold = generator_threshold

        self.optimizerG = optim.Adam([{'params' : self.netE.parameters()},
                         {'params' : self.netG.parameters()}], lr=lr, betas=(beta,0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta, 0.999))
    
   
    @property
    def log_dir(self):
        return logger.get_snapshot_dir()

    def log_sum_exp(self, input):
        m, _ = torch.max(input, dim=1, keepdim=True)
        input0 = input - m
        m.squeeze()
        return m + torch.log(torch.sum(torch.exp(input0), dim=1))

    def noise(self, size, num_epochs, epoch):
        return torch.Tensor(size).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs).to(self.device)

    def fixed_noise(self, b_size):
        return torch.randn(b_size, self.latent_size, 1, 1, device=self.device)

    def train_epoch(self, dataloader, epoch, num_epochs, get_data = id):
        for i, data in enumerate(dataloader, 0):
            #import ipdb; ipdb.set_trace()
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            data = get_data(data)
            real_d = data.to(self.device).float()
            b_size = real_d.size(0)

            real_label = torch.ones(b_size, device = self.device)
            fake_label = torch.zeros(b_size, device = self.device) 

            noise1 = self.noise(data.size(), num_epochs, epoch)
            noise2 = self.noise(data.size(), num_epochs, epoch)

            fake_z = self.fixed_noise(b_size)
            fake_d = self.netG(fake_z)
            # Encoder
            #real_z, _, _, _ = self.netE(real_d)
            real_z = torch.zeros([b_size, self.latent_size*2], device = self.device)
            real_z = real_z.view(b_size, -1)

            mu, log_sigma = real_z[:, :self.latent_size], real_z[:, self.latent_size:]
            sigma = torch.exp(log_sigma)
            epsilon = torch.randn(b_size, self.latent_size, device = self.device)
            output_z = mu + epsilon * sigma

            output_real, _ = self.netD(real_d + noise1, output_z.view(b_size, self.latent_size, 1, 1))
            output_fake, _ = self.netD(fake_d + noise2, fake_z)

            errD_real = self.criterion(output_real, real_label)
            errD_fake = self.criterion(output_fake, fake_label)
            errD = errD_real + errD_fake
            errG = self.criterion(output_fake, real_label) + self.criterion(output_real, fake_label)


            if errG.item() < self.generator_threshold:
                self.optimizerD.zero_grad()
                errD_real.backward(retain_graph=True)
                errD_fake.backward(retain_graph=True)
                self.optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.optimizerG.zero_grad()
            errG.backward()
            self.optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), output_real.mean().item(), output_fake.mean().item()))
            # Save Losses for plotting later
            self.G_losses.setdefault(epoch, []).append(errG.item())
            self.D_losses.setdefault(epoch, []).append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (self.iters % 200 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                #import ipdb; ipdb.set_trace()
                with torch.no_grad():
                    fake = self.netG(self.fixed_noise(64)).detach().cpu()
                sample = vutils.make_grid(fake, padding=2, normalize=True)
                self.img_list.append(sample)
                self.dump_samples("sample " + str(epoch), self.iters, sample)
                self.dump_samples("real " + str(epoch), self.iters, vutils.make_grid(real_d.cpu(), padding=2, normalize=True))

            self.iters += 1

    def dump_samples(self, epoch, iters, sample):
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.imshow(np.transpose(sample,(1,2,0)))
        save_dir = osp.join(self.log_dir, str(epoch) + '-' + str(iters) + '.png')
        plt.savefig(save_dir)

    def get_stats(self, epoch):
        stats = OrderedDict()
        stats["epoch"] = epoch
        stats["Generator Loss"] = np.mean(self.G_losses[epoch])
        stats["Discriminator Loss"] = np.mean(self.D_losses[epoch])
        return stats

    def get_G_losses(self):
        return self.G_losses

    def get_D_losses(self):
        return self.D_losses    

    def get_Encoder(self):
        return self.netE

    def get_Generator(self):
        return self.netG

    def get_Discriminator(self):
        return self.netD

    def get_img_list(self):
        return self.img_list
