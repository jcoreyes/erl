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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from railrl.core.loss import LossFunction
from railrl.core import logger
from torchvision.utils import save_image

class DCGANTrainer():

    def __init__(self, model, ngpu, lr, beta, nc, nz, ngf, ndf):
        self.Generator = model[0]
        self.Discriminator = model[1]
        self.img_list = []
        self.G_losses = {}
        self.D_losses = {}
        self.iters = 0
        self.real_label = 1
        self.fake_label = 0
        self.criterion = nn.BCELoss()

        
        self.ngpu = ngpu
        self.lr = lr
        self.beta = beta
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf

        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        self.netG = self.Generator(ngpu, nz, nc, ngf).to(self.device)
        self.netD = self.Discriminator(ngpu, nc, ndf).to(self.device)

        if (self.device.type == 'cuda') and (ngpu > 1):
            self.netG = nn.DataParallel(self.netG, list(range(self.ngpu)))

        self.netG.apply(self.weights_init)
        
        if (self.device.type == 'cuda') and (ngpu > 1):
            self.netD = nn.DataParallel(self.netD, list(range(ngpu)))

        self.netD.apply(self.weights_init)

        self.fixed_noise = torch.randn(64, nz, 1, 1, device=self.device)

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta, 0.999))
   
    @property
    def log_dir(self):
        return logger.get_snapshot_dir()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0,0.02)
            nn.init.constant_(m.bias.data, 0)

    def train_epoch(self, dataloader, epoch, num_epochs, get_data = lambda d: d):
        for i, data in enumerate(dataloader, 0):
            #import ipdb; ipdb.set_trace()
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            self.netD.zero_grad()
            real_cpu = get_data(data).to(self.device).float()
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), self.real_label, device=self.device)
            output = self.netD(real_cpu).view(-1)
            errD_real = self.criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
            fake = self.netG(noise)
            label.fill_(self.fake_label)
            output = self.netD(fake.detach()).view(-1)
            errD_fake = self.criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            self.optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.netG.zero_grad()
            label.fill_(self.real_label)  
            output = self.netD(fake).view(-1)
            errG = self.criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            self.optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            self.G_losses.setdefault(epoch, []).append(errG.item())
            self.D_losses.setdefault(epoch, []).append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (self.iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = self.netG(self.fixed_noise).detach().cpu()
                sample = vutils.make_grid(fake, padding=2, normalize=True)
                self.img_list.append(sample)
                self.dump_samples(epoch, self.iters, sample)

            self.iters += 1

    def dump_samples(self, epoch, iters, sample):
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.imshow(np.transpose(sample,(1,2,0)))
        save_dir = osp.join(self.log_dir, 's' + str(epoch) + '-' + str(iters) + '.png')
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

    def get_Generator(self):
        return self.Generator

    def get_Discriminator(self):
        return self.Discriminator

    def get_img_list(self):
        return self.img_list
