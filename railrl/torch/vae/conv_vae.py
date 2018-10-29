# Adapted from pytorch examples

from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.distributions import Normal
from torch.nn import functional as F
from torchvision.utils import save_image

from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.misc.ml_util import ConstantSchedule
from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler
from railrl.core import logger
import os.path as osp
import numpy as np
from multiworld.core.image_env import normalize_image
from railrl.torch.core import PyTorchModule
from railrl.core.serializable import Serializable

def inv_gaussian_p_x_np_to_np(model, data):
    ''' Assumes data is normalized images'''
    imgs = ptu.from_numpy(data)
    latents, mus, logvar, stds = get_encoding_and_suff_stats(model, imgs)
    true_prior = Normal(ptu.zeros(1), ptu.ones(1))
    vae_dist = Normal(mus, stds)
    log_p_z = true_prior.log_prob(latents).sum(dim=2)
    log_q_z_given_x = vae_dist.log_prob(latents).sum(dim=2)
    _, dec_mu, dec_var = model.decode_full(latents)
    dec_mu = dec_mu.view(dec_mu.shape[0] // latents.shape[1], latents.shape[1], dec_mu.shape[1])
    dec_var = dec_var.view(dec_var.shape[0] // latents.shape[1], latents.shape[1], dec_var.shape[1])
    decoder_dist = Normal(dec_mu, dec_var.pow(.5))
    log_d_x_given_z = decoder_dist.log_prob(imgs).sum(dim=2)
    return compute_inv_p_x_given_log_space_values(log_p_z, log_q_z_given_x, log_d_x_given_z)

def inv_p_bernoulli_x_np_to_np(model, data):
    ''' Assumes data is normalized images'''
    imgs = ptu.from_numpy(data)
    latents, mus, logvar, stds = get_encoding_and_suff_stats(model, imgs)
    true_prior = Normal(ptu.zeros(1), ptu.ones(1))
    vae_dist = Normal(mus, stds)
    log_p_z = true_prior.log_prob(latents).sum(dim=2)
    log_q_z_given_x = vae_dist.log_prob(latents).sum(dim=2)
    decoded = model.decode(latents)
    decoded = decoded.view(decoded.shape[0]//latents.shape[1], latents.shape[1], decoded.shape[1])
    imgs = imgs.view(imgs.shape[0], 1, imgs.shape[1])
    log_d_x_given_z = torch.log(imgs * decoded + (1 - imgs) * (1 - decoded) + 1e-8).sum(dim=2)
    inv_p_x_shifted = compute_inv_p_x_given_log_space_values(log_p_z, log_q_z_given_x, log_d_x_given_z)
    return inv_p_x_shifted

def compute_inv_p_x_given_log_space_values(log_p_z, log_q_z_given_x, log_d_x_given_z):
    log_p_x = log_p_z - log_q_z_given_x + log_d_x_given_z
    log_p_x = ((log_p_x - log_p_x.mean(dim=0)) / (log_p_x.std(dim=0)+1e-8)).mean(dim=1) # averages to gather all the samples num_latents_sampled
    log_inv_root_p_x = -1 / 2 * log_p_x
    log_inv_p_x_prime = log_inv_root_p_x - log_inv_root_p_x.max()
    inv_p_x_shifted = ptu.get_numpy(log_inv_p_x_prime.exp())
    return inv_p_x_shifted

def get_encoding_and_suff_stats(model, imgs):
    mu, logvar = model.encode(imgs)
    mu = mu.view((mu.size()[0], 1, mu.size()[1]))
    stds = (0.5 * logvar).exp()
    stds = stds.view(stds.size()[0], 1, stds.size()[1])
    epsilon = ptu.randn((mu.size()[0], model.num_latents_to_sample, mu.size()[1]))
    if ptu.gpu_enabled():
        epsilon = epsilon.cuda()
    latents = epsilon * stds + mu
    return latents, mu, logvar, stds

class ConvVAETrainer(Serializable):
    def __init__(
            self,
            train_dataset,
            test_dataset,
            model,
            batch_size=128,
            log_interval=0,
            beta=0.5,
            beta_schedule=None,
            lr=None,
            do_scatterplot=False,
            normalize=False,
            mse_weight=0.1,
            is_auto_encoder=False,
            background_subtract=False,
            linearity_weight=0.0,
            use_linear_dynamics=False,
            use_parallel_dataloading=True,
            train_data_workers=2,
            skew_dataset=False,
            skew_config=None,
            mean_squared_error_loss=False,
            gaussian_decoder_loss=False,
    ):
        self.quick_init(locals())
        if skew_config is None:
            skew_config = {}
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.beta = beta
        if is_auto_encoder:
            self.beta = 0
        if lr is None:
            if is_auto_encoder:
                lr = 1e-2
            else:
                lr = 1e-3
        self.beta_schedule = beta_schedule
        if self.beta_schedule is None or is_auto_encoder:
            self.beta_schedule = ConstantSchedule(self.beta)
        self.imsize = model.imsize
        self.do_scatterplot = do_scatterplot

        model.to(ptu.device)

        self.model = model
        self.representation_size = model.representation_size
        self.input_channels = model.input_channels
        self.imlength = model.imlength

        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        assert self.train_dataset.dtype == np.uint8
        assert self.test_dataset.dtype == np.uint8
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.use_parallel_dataloading = use_parallel_dataloading
        self.train_data_workers = train_data_workers
        self.skew_dataset = skew_dataset
        self.skew_config = skew_config
        self.mean_squared_error_loss = mean_squared_error_loss
        self.gaussian_decoder_loss=gaussian_decoder_loss
        if use_parallel_dataloading:
            self.train_dataset_pt = ImageDataset(
                train_dataset,
                should_normalize=True
            )
            self.test_dataset_pt = ImageDataset(
                test_dataset,
                should_normalize=True
            )

            if self.skew_dataset:
                self._train_weights = self._compute_train_weights()
                base_sampler = InfiniteWeightedRandomSampler(
                    self.train_dataset, self._train_weights
                )
            else:
                self._train_weights = None
                base_sampler = InfiniteRandomSampler(self.train_dataset)
            self.train_dataloader = DataLoader(
                self.train_dataset_pt,
                sampler=InfiniteRandomSampler(self.train_dataset),
                batch_size=batch_size,
                drop_last=False,
                num_workers=train_data_workers,
                pin_memory=True,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset_pt,
                sampler=InfiniteRandomSampler(self.test_dataset),
                batch_size=batch_size,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )
            self.train_dataloader = iter(self.train_dataloader)
            self.test_dataloader = iter(self.test_dataloader)

        self.normalize = normalize
        self.mse_weight = mse_weight
        self.background_subtract = background_subtract

        if self.normalize or self.background_subtract:
            self.train_data_mean = np.mean(self.train_dataset, axis=0)
            self.train_data_mean = normalize_image(
                np.uint8(self.train_data_mean)
            )
        self.linearity_weight = linearity_weight
        self.use_linear_dynamics = use_linear_dynamics
        self.vae_logger_stats_for_rl = {}
        self._extra_stats_to_log = None

    def get_dataset_stats(self, data):
        torch_input = ptu.np_to_var(normalize_image(data))
        mus, log_vars = self.model.encode(torch_input)
        mus = ptu.get_numpy(mus)
        mean = np.mean(mus, axis=0)
        std = np.std(mus, axis=0)
        return mus, mean, std

    def update_train_weights(self):
        # TODO: update the weights of the sampler rather than recreating loader
        if self.skew_dataset:
            self._train_weights = self._compute_train_weights()
            sampler = InfiniteWeightedRandomSampler(self.train_dataset, self._train_weights)
            self.train_dataloader.sampler = sampler
            self.train_dataloader = iter(self.train_dataloader)


    def _compute_train_weights(self):
        method = self.skew_config.get('method', 'squared_error')
        power = self.skew_config.get('power', 1)
        data = self.train_dataset
        if method == 'squared_error':
            return self._reconstruction_squared_error_np_to_np(
                data,
            ) ** power
        elif method == 'kl':
            return self._kl_np_to_np(data) ** power
        elif method == 'inv_gaussian_p_x':
            data = normalize_image(data)
            inv_gaussian_p_x = inv_gaussian_p_x_np_to_np(self.model, data) ** power
            return inv_gaussian_p_x
        elif method == 'inv_bernoulli_p_x':
            data = normalize_image(data)
            inv_bernoulli_p_x = inv_p_bernoulli_x_np_to_np(self.model, data) ** power
            return inv_bernoulli_p_x
        else:
            raise NotImplementedError('Method {} not supported'.format(method))


    def _kl_np_to_np(self, np_imgs):
        torch_input = ptu.np_to_var(normalize_image(np_imgs))
        mu, log_var = self.model.encode(torch_input)
        return ptu.get_numpy(
            - torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )

    def _reconstruction_squared_error_np_to_np(self, np_imgs):
        torch_input = ptu.np_to_var(normalize_image(np_imgs))
        recons, *_ = self.model(torch_input)
        error = torch_input - recons
        return ptu.get_numpy((error ** 2).sum(dim=1))

    def set_vae(self, vae):
        self.model = vae
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def get_batch(self, train=True):
        if self.use_parallel_dataloading:
            if not train:
                dataloader = self.test_dataloader
            else:
                dataloader = self.train_dataloader
            samples = next(dataloader).to(ptu.device)
            return samples

        dataset = self.train_dataset if train else self.test_dataset
        ind = np.random.randint(0, len(dataset), self.batch_size)
        samples = normalize_image(dataset[ind, :])
        if self.normalize:
            samples = ((samples - self.train_data_mean) + 1) / 2
        if self.background_subtract:
            samples = samples - self.train_data_mean
        return ptu.from_numpy(samples)


    def get_debug_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        X, Y = dataset
        ind = np.random.randint(0, Y.shape[0], self.batch_size)
        X = X[ind, :]
        Y = Y[ind, :]
        return ptu.from_numpy(X), ptu.from_numpy(Y)

    def compute_bernoulli_log_prob(self, recon_x, x):
        # Divide by batch_size rather than setting size_average=True because
        # otherwise the averaging will also happen across dim 1 (the
        # pixels)
        return -1* F.binary_cross_entropy(
            recon_x,
            x.narrow(start=0, length=self.imlength,
                     dim=1).contiguous().view(-1, self.imlength),
            size_average=False,
        ) / self.batch_size

    def kl_divergence(self, recon_x, x, mu, logvar):
        return - torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def state_linearity_loss(self, obs, next_obs, actions):
        latent_obs = self.model.encode(obs)[0]
        latent_next_obs = self.model.encode(next_obs)[0]
        action_obs_pair = torch.cat([latent_obs, actions], dim=1)
        prediction = self.model.linear_constraint_fc(action_obs_pair)
        return torch.norm(prediction - latent_next_obs) ** 2 / self.batch_size

    def compute_gaussian_log_prob(self, input, dec_mu, dec_var):
        dec_mu = dec_mu.view(-1, self.input_channels*self.imsize**2)
        dec_var = dec_var.view(-1, self.input_channels*self.imsize**2)
        decoder_dist = Normal(dec_mu, dec_var.pow(0.5))
        input = input.view(-1, self.input_channels*self.imsize**2)
        log_probs = decoder_dist.log_prob(input)
        vals = log_probs.sum(dim=1, keepdim=True)
        return vals.mean()

    def logprob_and_stats(self, next_obs):
        if self.gaussian_decoder_loss:
            latents, mu, logvar, stds = get_encoding_and_suff_stats(self.model, next_obs)
            recon_batch, dec_mu, dec_var = self.model.decode_full(latents)
            log_prob = self.compute_gaussian_log_prob(next_obs, dec_mu, dec_var)
        else:
            recon_batch, mu, logvar = self.model(next_obs)
            log_prob = self.compute_bernoulli_log_prob(recon_batch, next_obs)
        return log_prob, recon_batch, next_obs, mu, logvar

    def train_epoch(self, epoch, sample_batch=None, batches=100, from_rl=False):
        self.model.train()
        losses = []
        log_probs = []
        kles = []
        linear_losses = []
        beta = float(self.beta_schedule.get_value(epoch))
        for batch_idx in range(batches):
            if sample_batch is not None:
                data = sample_batch(self.batch_size)
                # obs = data['obs']
                next_obs = data['next_obs']
                # actions = data['actions']
            else:
                next_obs = self.get_batch()
                obs = None
                actions = None
            self.optimizer.zero_grad()
            log_prob, recon_batch, next_obs, mu, logvar = self.logprob_and_stats(next_obs)
            kle = self.kl_divergence(recon_batch, next_obs, mu, logvar)

            if self.use_linear_dynamics:
                linear_dynamics_loss = self.state_linearity_loss(
                    obs, next_obs, actions
                )
                loss = log_prob + beta * kle + self.linearity_weight * linear_dynamics_loss
                linear_losses.append(linear_dynamics_loss.data[0])
            else:
                loss = -1*log_prob + beta * kle
            loss.backward()

            losses.append(loss.item())
            log_probs.append(log_prob.item())
            kles.append(kle.item())

            self.optimizer.step()
            if self.log_interval and batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(next_obs)))

        if from_rl:
            self.vae_logger_stats_for_rl['Train VAE Epoch'] = epoch
            self.vae_logger_stats_for_rl['Train VAE Log Prob'] = np.mean(log_probs)
            self.vae_logger_stats_for_rl['Train VAE KL'] = np.mean(kles)
            self.vae_logger_stats_for_rl['Train VAE Loss'] = np.mean(losses)
            if self.use_linear_dynamics:
                self.vae_logger_stats_for_rl['Train VAE Linear_loss'] = \
                    np.mean(linear_losses)
        else:
            logger.record_tabular("train/epoch", epoch)
            logger.record_tabular("train/Log Prob", np.mean(log_probs))
            logger.record_tabular("train/KL", np.mean(kles))
            logger.record_tabular("train/loss", np.mean(losses))
            if self.use_linear_dynamics:
                logger.record_tabular("train/linear_loss",
                                      np.mean(linear_losses))

    def test_epoch(
            self,
            epoch,
            save_reconstruction=True,
            save_scatterplot=True,
            save_vae=True,
            from_rl=False,
    ):
        self.model.eval()
        losses = []
        log_probs = []
        kles = []
        zs = []
        beta = float(self.beta_schedule.get_value(epoch))
        for batch_idx in range(10):
            next_obs = self.get_batch(train=False)
            log_prob, recon_batch, next_obs, mu, logvar = self.logprob_and_stats(next_obs)
            kle = self.kl_divergence(recon_batch, next_obs, mu, logvar)
            loss = -1*log_prob + beta * kle

            z_data = ptu.get_numpy(mu.cpu())
            for i in range(len(z_data)):
                zs.append(z_data[i, :])
            losses.append(loss.item())
            log_probs.append(log_prob.item())
            kles.append(kle.item())

            if batch_idx == 0 and save_reconstruction:
                n = min(next_obs.size(0), 8)
                comparison = torch.cat([
                    next_obs[:n].narrow(start=0, length=self.imlength, dim=1)
                    .contiguous().view(
                        -1, self.input_channels, self.imsize, self.imsize
                    ),
                    recon_batch.view(
                        self.batch_size,
                        self.input_channels,
                        self.imsize,
                        self.imsize,
                    )[:n]
                ])
                save_dir = osp.join(logger.get_snapshot_dir(),
                                    'r%d.png' % epoch)
                save_image(comparison.data.cpu(), save_dir, nrow=n)

        zs = np.array(zs)
        self.model.dist_mu = zs.mean(axis=0)
        self.model.dist_std = zs.std(axis=0)
        if self.do_scatterplot and save_scatterplot:
            self.plot_scattered(np.array(zs), epoch)


        if self.skew_dataset:
            train_weight_mean = np.mean(self._train_weights)
            num_above_avg = np.sum(np.where(self._train_weights >= train_weight_mean, 1, 0))
            logger.record_tabular("% train weights above average", num_above_avg/self.train_dataset.shape[0])

        if from_rl:
            self.vae_logger_stats_for_rl['Test VAE Epoch'] = epoch
            self.vae_logger_stats_for_rl['Test VAE Log Prob'] = np.mean(log_probs)
            self.vae_logger_stats_for_rl['Test VAE KL'] = np.mean(kles)
            self.vae_logger_stats_for_rl['Test VAE loss'] = np.mean(losses)
            self.vae_logger_stats_for_rl['VAE Beta'] = beta
        else:
            for key, value in self.debug_statistics().items():
                logger.record_tabular(key, value)

            logger.record_tabular("test/Log Prob", np.mean(log_probs))
            logger.record_tabular("test/KL", np.mean(kles))
            logger.record_tabular("test/loss", np.mean(losses))
            logger.record_tabular("beta", beta)
            logger.dump_tabular()
            if save_vae:
                logger.save_itr_params(epoch, self.model)  # slow...
        # logdir = logger.get_snapshot_dir()
        # filename = osp.join(logdir, 'params.pkl')
        # torch.save(self.model, filename)

    def debug_statistics(self):
        """
        Given an image $$x$$, samples a bunch of latents from the prior
        $$z_i$$ and decode them $$\hat x_i$$.
        Compare this to $$\hat x$$, the reconstruction of $$x$$.
        Ideally
         - All the $$\hat x_i$$s do worse than $$\hat x$$ (makes sure VAE
           isn’t ignoring the latent)
         - Some $$\hat x_i$$ do better than other $$\hat x_i$$ (tests for
           coverage)
        """
        debug_batch_size = 64
        data = self.get_batch(train=False)
        recon_batch, mu, logvar = self.model(data)
        img = data[0]
        recon_mse = ((recon_batch[0] - img)**2).mean().view(-1)
        img_repeated = img.expand((debug_batch_size, img.shape[0]))

        samples = ptu.randn(debug_batch_size, self.representation_size)
        random_imgs = self.model.decode(samples)
        random_mses = (random_imgs - img_repeated) ** 2
        mse_improvement = ptu.get_numpy(random_mses.mean(dim=1) - recon_mse)
        stats = create_stats_ordered_dict(
            'debug/MSE improvement over random',
            mse_improvement,
        )
        stats.update(create_stats_ordered_dict(
            'debug/MSE of random decoding',
            ptu.get_numpy(random_mses),
        ))
        stats['debug/MSE of reconstruction'] = ptu.get_numpy(
            recon_mse
        )[0]
        if self.skew_dataset:
            stats.update(create_stats_ordered_dict(
                'train weight',
                self._train_weights
            ))
        return stats

    def dump_samples(self, epoch):
        self.model.eval()
        sample = ptu.randn(64, self.representation_size)
        sample = self.model.decode(sample).cpu()
        save_dir = osp.join(logger.get_snapshot_dir(), 's%d.png' % epoch)
        save_image(
            sample.data.view(64, self.input_channels, self.imsize, self.imsize),
            save_dir
        )

    def dump_sampling_histogram(self, epoch):
        import matplotlib.pyplot as plt
        if self._train_weights is None:
            self._train_weights = self._compute_train_weights()
        weights = torch.from_numpy(self._train_weights)
        samples = torch.multinomial(
            weights, len(weights), replacement=True
        )
        plt.clf()
        n, bins, patches = plt.hist(samples, bins=np.arange(0, len(weights)))
        save_file = osp.join(logger.get_snapshot_dir(), 'hist{}.png'.format(
            epoch))
        plt.savefig(save_file)
        data_save_file = osp.join(logger.get_snapshot_dir(), 'hist_data.txt')
        with open(data_save_file, 'a') as f:
            f.write(str(list(zip(bins, n))))
            f.write('\n')

    def dump_best_reconstruction(self, epoch, num_shown=4):
        idx_and_weights = self._get_sorted_idx_and_train_weights()
        idxs = [i for i, _ in idx_and_weights[:num_shown]]
        self._dump_imgs_and_reconstructions(idxs, 'best{}.png'.format(epoch))

    def dump_worst_reconstruction(self, epoch, num_shown=4):
        idx_and_weights = self._get_sorted_idx_and_train_weights()
        idx_and_weights = idx_and_weights[::-1]
        idxs = [i for i, _ in idx_and_weights[:num_shown]]
        self._dump_imgs_and_reconstructions(idxs, 'worst{}.png'.format(epoch))

    def _dump_imgs_and_reconstructions(self, idxs, filename):
        imgs = []
        recons = []
        for i in idxs:
            img_np = self.train_dataset[i]
            img_torch = ptu.np_to_var(normalize_image(img_np))
            recon, *_ = self.model(img_torch)

            img = img_torch.view(self.input_channels, self.imsize, self.imsize)
            rimg = recon.view(self.input_channels, self.imsize, self.imsize)
            imgs.append(img)
            recons.append(rimg)
        all_imgs = torch.stack(imgs + recons)
        save_file = osp.join(logger.get_snapshot_dir(), filename)
        save_image(
            all_imgs.data,
            save_file,
            nrow=4,
        )

    def _get_sorted_idx_and_train_weights(self):
        if self._train_weights is None:
            self._train_weights = self._compute_train_weights()
        idx_and_weights = zip(range(len(self._train_weights)),
                              self._train_weights)
        return sorted(idx_and_weights, key=lambda x: x[0])

    def plot_scattered(self, z, epoch):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.log(__file__ + ": Unable to load matplotlib. Consider "
                                  "setting do_scatterplot to False")
            return
        dim_and_stds = [(i, np.std(z[:, i])) for i in range(z.shape[1])]
        dim_and_stds = sorted(
            dim_and_stds,
            key=lambda x: x[1]
        )
        dim1 = dim_and_stds[-1][0]
        dim2 = dim_and_stds[-2][0]
        plt.figure(figsize=(8, 8))
        plt.scatter(z[:, dim1], z[:, dim2], marker='o', edgecolor='none')
        if self.model.dist_mu is not None:
            x1 = self.model.dist_mu[dim1:dim1 + 1]
            y1 = self.model.dist_mu[dim2:dim2 + 1]
            x2 = (
                self.model.dist_mu[dim1:dim1 + 1]
                + self.model.dist_std[dim1:dim1 + 1]
            )
            y2 = (
                self.model.dist_mu[dim2:dim2 + 1]
                + self.model.dist_std[dim2:dim2 + 1]
            )
        plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
        axes = plt.gca()
        axes.set_xlim([-6, 6])
        axes.set_ylim([-6, 6])
        axes.set_title('dim {} vs dim {}'.format(dim1, dim2))
        plt.grid(True)
        save_file = osp.join(logger.get_snapshot_dir(), 'scatter%d.png' % epoch)
        plt.savefig(save_file)


class ConvVAESmall(PyTorchModule):
    def __init__(
            self,
            representation_size,
            init_w=1e-3,
            input_channels=1,
            imsize=48,
            hidden_init=ptu.fanin_init,
            encoder_activation=identity,
            decoder_activation=identity,
            min_variance=1e-3,
            state_size=0,
            num_latents_to_sample=1,
    ):
        self.save_init_params(locals())
        super().__init__()
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))

        self.representation_size = representation_size
        self.hidden_init = hidden_init
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.num_latents_to_sample=num_latents_to_sample
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize ** 2 * self.input_channels
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)
        self.relu = nn.ReLU()
        self.init_w = init_w
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.kernel_out = 64
        self.conv_output_dim = self.kernel_out * 9
        self.fc1 = nn.Linear(self.conv_output_dim, representation_size)
        self.fc2 = nn.Linear(self.conv_output_dim, representation_size)
        self.fc3 = nn.Linear(representation_size, self.conv_output_dim)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2)
        self.conv6 = nn.ConvTranspose2d(16, input_channels, kernel_size=6,
                                        stride=3)
        self.epoch = 0
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        self.hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        self.hidden_init(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        self.hidden_init(self.conv4.weight)
        self.conv4.bias.data.fill_(0)
        self.hidden_init(self.conv5.weight)
        self.conv5.bias.data.fill_(0)
        self.hidden_init(self.conv6.weight)
        self.conv6.bias.data.fill_(0)

        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

        # self.fc4.weight.data.uniform_(-init_w, init_w)
        # self.fc4.bias.data.uniform_(-init_w, init_w)

    def encode(self, input):
        input = input.view(-1, self.imlength)
        conv_input = input
        x = conv_input.contiguous().view(-1, self.input_channels, self.imsize,
                                         self.imsize)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        h = x.view(-1, self.conv_output_dim)  # flatten
        # h = self.relu(self.fc4(h))
        mu = self.encoder_activation(self.fc1(h))
        if self.log_min_variance is None:
            logvar = self.encoder_activation(self.fc2(h))
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h = h3.view(-1, self.kernel_out, 3, 3)
        x = F.relu(self.conv4(h))
        x = F.relu(self.conv5(x))
        x = self.conv6(x).view(-1,
                               self.imsize * self.imsize * self.input_channels)

        return self.decoder_activation(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

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

class ConvVAESmallDouble(PyTorchModule):
    def __init__(
            self,
            representation_size,
            init_w=1e-3,
            input_channels=1,
            imsize=48,
            hidden_init=ptu.fanin_init,
            encoder_activation=identity,
            decoder_activation=identity,
            min_variance=1e-3,
            state_size=0,
            unit_variance=False,
            is_auto_encoder=False,
            variance_scaling=1,
    ):
        self.save_init_params(locals())
        super().__init__()
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))

        self.representation_size = representation_size
        self.variance_scaling=variance_scaling
        self.hidden_init = hidden_init
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize ** 2 * self.input_channels
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)
        self.unit_variance = unit_variance
        self.relu = nn.ReLU()
        self.init_w = init_w
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.kernel_out = 64
        self.conv_output_dim = self.kernel_out * 9
        self.fc1 = nn.Linear(self.conv_output_dim, representation_size)
        self.fc2 = nn.Linear(self.conv_output_dim, representation_size)
        self.fc3 = nn.Linear(representation_size, self.conv_output_dim)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2)
        self.dec_mu = nn.ConvTranspose2d(16, input_channels, kernel_size=6,
                                         stride=3)
        self.dec_var = nn.ConvTranspose2d(16, input_channels, kernel_size=6,
                                          stride=3)
        self.epoch = 0
        self.is_auto_encoder=is_auto_encoder
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        self.hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        self.hidden_init(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        self.hidden_init(self.conv4.weight)
        self.conv4.bias.data.fill_(0)
        self.hidden_init(self.conv5.weight)
        self.conv5.bias.data.fill_(0)
        self.hidden_init(self.dec_mu.weight)
        self.dec_mu.bias.data.fill_(0)
        self.hidden_init(self.dec_var.weight)
        self.dec_var.bias.data.fill_(0)

        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

        # self.fc4.weight.data.uniform_(-init_w, init_w)
        # self.fc4.bias.data.uniform_(-init_w, init_w)

    def encode(self, input):
        input = input.view(-1, self.imlength)
        conv_input = input
        x = conv_input.contiguous().view(-1, self.input_channels, self.imsize,
                                         self.imsize)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        h = x.view(-1, self.conv_output_dim)  # flatten
        # h = self.relu(self.fc4(h))
        mu = self.encoder_activation(self.fc1(h))
        if self.log_min_variance is None:
            logvar = self.encoder_activation(self.fc2(h))
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h = h3.view(-1, self.kernel_out, 3, 3)
        x = F.relu(self.conv4(h))
        h = F.relu(self.conv5(x))
        mu = self.dec_mu(h).view(-1,
                                self.imsize * self.imsize * self.input_channels)
        return self.decoder_activation(mu)

    def decode_full(self, z):
        h3 = self.relu(self.fc3(z))
        h = h3.view(-1, self.kernel_out, 3, 3)
        x = F.relu(self.conv4(h))
        h = F.relu(self.conv5(x))
        mu = self.dec_mu(h).view(-1,
                                 self.imsize * self.imsize * self.input_channels)
        logvar = self.dec_var(h).view(-1,
                                   self.imsize * self.imsize * self.input_channels)
        var = logvar.exp()
        if self.unit_variance:
            var = torch.ones_like(var)
        return self.decoder_activation(mu), self.decoder_activation(mu), self.decoder_activation(var)*self.variance_scaling

    def forward(self, x):
        mu, logvar = self.encode(x)
        if self.is_auto_encoder:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

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


class ConvVAE(PyTorchModule):
    def __init__(
            self,
            representation_size,
            init_w=1e-3,
            input_channels=1,
            imsize=84,
            added_fc_size=0,
            hidden_init=ptu.fanin_init,
            output_activation=identity,
            min_variance=1e-4,
            use_min_variance=True,
            state_size=0,
            action_dim=None,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.representation_size = representation_size
        self.hidden_init = hidden_init
        self.output_activation = output_activation
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize ** 2 * self.input_channels
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.added_fc_size = added_fc_size
        self.init_w = init_w

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=3)
        self.bn3 = nn.BatchNorm2d(32)

        # self.conv_output_dim = 1568 # kernel 2
        self.conv_output_dim = 128  # kernel 3

        # self.hidden = nn.Linear(self.conv_output_dim + added_fc_size, representation_size)

        self.fc1 = nn.Linear(self.conv_output_dim, representation_size)
        self.fc2 = nn.Linear(self.conv_output_dim, representation_size)

        self.fc3 = nn.Linear(representation_size, self.conv_output_dim)
        self.conv4 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=6, stride=3)
        self.conv6 = nn.ConvTranspose2d(16, input_channels, kernel_size=6,
                                        stride=3)

        if action_dim is not None:
            self.linear_constraint_fc = \
                nn.Linear(
                    self.representation_size + action_dim,
                    self.representation_size
                )
        else:
            self.linear_constraint_fc = None

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        self.hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        self.hidden_init(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        self.hidden_init(self.conv4.weight)
        self.conv4.bias.data.fill_(0)
        self.hidden_init(self.conv5.weight)
        self.conv5.bias.data.fill_(0)
        self.hidden_init(self.conv6.weight)
        self.conv6.bias.data.fill_(0)

        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def encode(self, input):
        input = input.view(-1, self.imlength + self.added_fc_size)
        conv_input = input.narrow(start=0, length=self.imlength, dim=1)

        x = conv_input.contiguous().view(-1, self.input_channels, self.imsize,
                                         self.imsize)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        h = x.view(-1, 128)  # flatten
        if self.added_fc_size != 0:
            fc_input = input.narrow(start=self.imlength, length=self.added_fc_size, dim=1)
            h = torch.cat((h, fc_input), dim=1)
        mu = self.output_activation(self.fc1(h))
        if self.log_min_variance is None:
            logvar = self.output_activation(self.fc2(h))
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h = h3.view(-1, 32, 2, 2)
        x = F.relu(self.conv4(h))
        x = F.relu(self.conv5(x))
        x = self.conv6(x).view(-1, self.imsize*self.imsize*self.input_channels)
        return self.sigmoid(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

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


class ConvVAELarge(PyTorchModule):
    def __init__(
            self,
            representation_size,
            init_w=1e-3,
            input_channels=1,
            imsize=84,
            added_fc_size=0,
            hidden_init=ptu.fanin_init,
            output_activation=identity,
            min_variance=1e-4,
    ):
        self.save_init_params(locals())
        # TODO(mdalal2020): You probably want to fix this init call...
        super().__init__()
        self.representation_size = representation_size
        self.hidden_init = hidden_init
        self.output_activation = output_activation
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize ** 2 * self.input_channels
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.added_fc_size = added_fc_size
        self.init_w = init_w

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=3)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, stride=3)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn6 = nn.BatchNorm2d(32)

        self.conv_output_dim = 128

        self.fc1 = nn.Linear(self.conv_output_dim, representation_size)
        self.fc2 = nn.Linear(self.conv_output_dim, representation_size)

        self.fc3 = nn.Linear(representation_size, self.conv_output_dim)

        self.conv7 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2)
        self.conv8 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3)
        self.conv9 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3)
        self.conv10 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1)
        self.conv11 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1)
        self.conv12 = nn.ConvTranspose2d(16, input_channels, kernel_size=6,
                                         stride=1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        self.hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        self.hidden_init(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        self.hidden_init(self.conv4.weight)
        self.conv4.bias.data.fill_(0)
        self.hidden_init(self.conv5.weight)
        self.conv5.bias.data.fill_(0)
        self.hidden_init(self.conv6.weight)
        self.conv6.bias.data.fill_(0)

        self.hidden_init(self.conv7.weight)
        self.conv7.bias.data.fill_(0)
        self.hidden_init(self.conv8.weight)
        self.conv8.bias.data.fill_(0)
        self.hidden_init(self.conv9.weight)
        self.conv9.bias.data.fill_(0)
        self.hidden_init(self.conv10.weight)
        self.conv10.bias.data.fill_(0)
        self.hidden_init(self.conv11.weight)
        self.conv11.bias.data.fill_(0)
        self.hidden_init(self.conv12.weight)
        self.conv12.bias.data.fill_(0)

        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

        self.hidden_init(self.fc3.weight)
        self.fc3.bias.data.fill_(0)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def encode(self, input):
        input = input.view(-1, self.imlength + self.added_fc_size)
        conv_input = input.narrow(start=0, length=self.imlength, dim=1)

        # batch_size = input.size(0)
        x = conv_input.contiguous().view(-1, self.input_channels, self.imsize,
                                         self.imsize)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        h = x.view(-1, 128)  # flatten
        if self.added_fc_size != 0:
            fc_input = input.narrow(start=self.imlength, length=self.added_fc_size, dim=1)
            h = torch.cat((h, fc_input), dim=1)
        mu = self.output_activation(self.fc1(h))
        if self.log_min_variance is None:
            logvar = self.output_activation(self.fc2(h))
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))
        return mu, logvar

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h = h3.view(-1, 32, 2, 2)
        x = F.relu(self.conv7(h))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = self.conv12(x).view(-1,
                                self.imsize * self.imsize * self.input_channels)
        return self.sigmoid(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

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

class AutoEncoder(ConvVAE):
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu
        return self.decode(z), mu, logvar


class SpatialVAE(ConvVAE):
    def __init__(
            self,
            representation_size,
            num_feat_points,
            *args,
            temperature=1.0,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(representation_size, *args, **kwargs)
        self.num_feat_points = num_feat_points
        self.conv3 = nn.Conv2d(32, self.num_feat_points, kernel_size=5,
                               stride=3)
        #        self.bn3 = nn.BatchNorm2d(32)

        test_mat = torch.zeros(1, self.input_channels, self.imsize, self.imsize)
        test_mat = self.conv1(test_mat)
        test_mat = self.conv2(test_mat)
        test_mat = self.conv3(test_mat)
        self.out_size = int(np.prod(test_mat.shape))

        self.spatial_fc = nn.Linear(
            2 * self.num_feat_points + self.added_fc_size, 64)

        # self.conv_output_dim = 1568 # kernel 2
        self.conv_output_dim = 128  # kernel 3

        self.fc1 = nn.Linear(64, representation_size)
        self.fc2 = nn.Linear(64, representation_size)

        self.init_weights(self.init_w)
        self.temperature = temperature

    def init_weights_spatial(self, init_w):
        self.hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        self.hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        self.hidden_init(self.conv3.weight)
        self.conv3.bias.data.fill_(0)

        self.hidden_init(self.spatial_fc.weight)
        self.spatial_fc.bias.data.fill_(0)
        self.spatial_fc.weight.data.uniform_(-init_w, init_w)
        self.spatial_fc.bias.data.uniform_(-init_w, init_w)

        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

    def encode(self, input):
        input = input.view(-1, self.imlength + self.added_fc_size)
        conv_input = input.narrow(start=0, length=self.imlength, dim=1)

        # batch_size = input.size(0)
        x = conv_input.contiguous().view(-1, self.input_channels, self.imsize,
                                         self.imsize)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        d = int((self.out_size // self.num_feat_points) ** (1 / 2))
        x = x.view(-1, self.num_feat_points, d * d)
        x = F.softmax(x / self.temperature, 2)
        x = x.view(-1, self.num_feat_points, d, d)

        maps_x = torch.sum(x, 2)
        maps_y = torch.sum(x, 3)

        weights = ptu.from_numpy(np.arange(d) / (d + 1))

        fp_x = torch.sum(maps_x * weights, 2)
        fp_y = torch.sum(maps_y * weights, 2)

        x = torch.cat([fp_x, fp_y], 1)
        h = x.view(-1, self.num_feat_points * 2)
        if self.added_fc_size != 0:
            fc_input = input.narrow(start=self.imlength, length=self.added_fc_size, dim=1)
            h = torch.cat((h, fc_input), dim=1)
        h = F.relu(self.spatial_fc(h))
        mu = self.output_activation(self.fc1(h))
        logvar = self.output_activation(self.fc2(h))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        return mu

class ImageDataset(Dataset):

    def __init__(self, images, should_normalize=True):
        super().__init__()
        self.dataset = images
        self.dataset_len = len(self.dataset)
        assert should_normalize == (images.dtype == np.uint8)
        self.should_normalize = should_normalize

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idxs):
        samples = self.dataset[idxs, :]
        if self.should_normalize:
            samples = normalize_image(samples)
        return np.float32(samples)


class InfiniteRandomSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.iter = iter(torch.randperm(len(self.data_source)).tolist())

    def __iter__(self):
        return self

    def __next__(self):
        try:
            idx = next(self.iter)
        except StopIteration:
            self.iter = iter(torch.randperm(len(self.data_source)).tolist())
            idx = next(self.iter)
        return idx

    def __len__(self):
        return 2 ** 62


class InfiniteWeightedRandomSampler(Sampler):

    def __init__(self, data_source, weights):
        assert len(data_source) == len(weights)
        assert len(weights.shape) == 1
        self.data_source = data_source
        # Always use CPU
        self._weights = torch.from_numpy(weights)
        self.iter = self._create_iterator()

    def update_weights(self, weights):
        self._weights = weights
        self.iter = self._create_iterator()

    def _create_iterator(self):
        return iter(
            torch.multinomial(
                self._weights, len(self._weights), replacement=True
            ).tolist()
        )

    def __iter__(self):
        return self

    def __next__(self):
        try:
            idx = next(self.iter)
        except StopIteration:
            self.iter = self._create_iterator()
            idx = next(self.iter)
        return idx

    def __len__(self):
        return 2 ** 62


if __name__ == "__main__":
    m = ConvVAE(2)
    for epoch in range(10):
        m.train_epoch(epoch)
        m.test_epoch(epoch)
        m.dump_samples(epoch)
