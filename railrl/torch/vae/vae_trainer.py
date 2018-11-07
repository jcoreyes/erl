from os import path as osp

import numpy as np
import torch
from torch import optim
from torch.distributions import Normal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from multiworld.core.image_env import normalize_image
from railrl.core import logger
from railrl.core.serializable import Serializable
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.misc.ml_util import ConstantSchedule
from railrl.torch import pytorch_util as ptu
from railrl.torch.data import (
    ImageDataset, InfiniteWeightedRandomSampler,
    InfiniteRandomSampler,
)

def inv_gaussian_p_x_np_to_np(model, data, num_latents_to_sample=1):
    ''' Assumes data is normalized images'''
    imgs = ptu.from_numpy(data)
    latent_distribution_params = model.encode(imgs)
    latents = model.rsample(latent_distribution_params, num_latents_to_sample=num_latents_to_sample)
    mus, logvars = latent_distribution_params
    stds = logvars.exp().pow(.5)
    true_prior = Normal(ptu.zeros(1), ptu.ones(1))
    vae_dist = Normal(mus, stds)
    log_p_z = true_prior.log_prob(latents).sum(dim=2)
    log_q_z_given_x = vae_dist.log_prob(latents).sum(dim=2)
    _, obs_distribution_params = model.decode(latents)
    dec_mu, dec_var = obs_distribution_params
    dec_mu = dec_mu.view(dec_mu.shape[0] // latents.shape[1], latents.shape[1], dec_mu.shape[1])
    dec_logvar = dec_logvar.view(dec_logvar.shape[0] // latents.shape[1], latents.shape[1], dec_logvar.shape[1])
    dec_var = dec_logvar.exp()
    decoder_dist = Normal(dec_mu, dec_var.pow(.5))
    imgs = imgs.view(imgs.shape[0], 1, imgs.shape[1])
    log_d_x_given_z = decoder_dist.log_prob(imgs)
    log_d_x_given_z = log_d_x_given_z.sum(dim=2)
    return compute_inv_p_x_given_log_space_values(log_p_z, log_q_z_given_x, log_d_x_given_z)


def inv_p_bernoulli_x_np_to_np(model, data, num_latents_to_sample=1):
    ''' Assumes data is normalized images'''
    imgs = ptu.from_numpy(data)
    latent_distribution_params = model.encode(imgs)
    latents = model.rsample(latent_distribution_params, num_latents_to_sample=num_latents_to_sample)
    mus, logvars = latent_distribution_params
    stds = logvars.exp().pow(.5)
    true_prior = Normal(ptu.zeros(1), ptu.ones(1))
    vae_dist = Normal(mus, stds)
    log_p_z = true_prior.log_prob(latents).sum(dim=2)
    log_q_z_given_x = vae_dist.log_prob(latents).sum(dim=2)
    decoded = model.decode(latents)[0]
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
            priority_function_kwargs=None,
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
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr)
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
        if priority_function_kwargs is None:
            self.priority_function_kwargs = dict()
        else:
            self.priority_function_kwargs = priority_function_kwargs
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
        torch_input = ptu.from_numpy(normalize_image(data))
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
        batch_size = 1024
        size = self.train_dataset.shape[0]
        next_idx = min(batch_size, size)
        cur_idx = 0
        weights = np.zeros(size)
        while cur_idx < self.train_dataset.shape[0]:
            idxs = np.arange(cur_idx, next_idx)
            data = self.train_dataset[idxs, :]
            if method == 'squared_error':
                weights[idxs] = self._reconstruction_squared_error_np_to_np(
                    data,
                    **self.priority_function_kwargs
                ) ** power
            elif method == 'kl':
                weights[idxs] = self._kl_np_to_np(data, **self.priority_function_kwargs) ** power
            elif method == 'inv_gaussian_p_x':
                data = normalize_image(data)
                weights[idxs] = inv_gaussian_p_x_np_to_np(self.model, data, **self.priority_function_kwargs) ** power
            elif method == 'inv_bernoulli_p_x':
                data = normalize_image(data)
                weights[idxs] = inv_p_bernoulli_x_np_to_np(self.model, data, **self.priority_function_kwargs) ** power
            else:
                raise NotImplementedError('Method {} not supported'.format(method))
            cur_idx = next_idx
            next_idx += batch_size
            next_idx = min(next_idx, size)
        return weights

    def _kl_np_to_np(self, np_imgs):
        torch_input = ptu.from_numpy(normalize_image(np_imgs))
        mu, log_var = self.model.encode(torch_input)
        return ptu.get_numpy(
            - torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )

    def _reconstruction_squared_error_np_to_np(self, np_imgs):
        torch_input = ptu.from_numpy(normalize_image(np_imgs))
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

    def state_linearity_loss(self, obs, next_obs, actions):
        latent_obs = self.model.encode(obs)[0]
        latent_next_obs = self.model.encode(next_obs)[0]
        action_obs_pair = torch.cat([latent_obs, actions], dim=1)
        prediction = self.model.linear_constraint_fc(action_obs_pair)
        return torch.norm(prediction - latent_next_obs) ** 2 / self.batch_size

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
            reconstructions, obs_distribution_params, latent_distribution_params = self.model(next_obs)
            log_prob = self.model.logprob(next_obs, obs_distribution_params)
            kle = self.model.kl_divergence(latent_distribution_params)

            if self.use_linear_dynamics:
                linear_dynamics_loss = self.state_linearity_loss(
                    obs, next_obs, actions
                )
                loss = -1*log_prob + beta * kle + self.linearity_weight * linear_dynamics_loss
                linear_losses.append(linear_dynamics_loss.data[0])
            else:
                loss = -1*log_prob + beta * kle

            # loss += .01*(self.max_logvar.sum()) -.01*(self.min_logvar.sum())
            self.optimizer.zero_grad()
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
            # logger.record_tabular("train/max_logvar_mean", self.max_logvar.mean().item())
            # logger.record_tabular("train/min_logvar_mean", self.min_logvar.mean().item())


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
            reconstructions, obs_distribution_params, latent_distribution_params = self.model(next_obs)
            log_prob = self.model.logprob(next_obs, obs_distribution_params)
            kle = self.model.kl_divergence(latent_distribution_params)
            loss = -1*log_prob + beta * kle

            encoder_mean = latent_distribution_params[0]
            z_data = ptu.get_numpy(encoder_mean.cpu())
            for i in range(len(z_data)):
                zs.append(z_data[i, :])
            losses.append(loss.item())
            log_probs.append(log_prob.item())
            kles.append(kle.item())

            # loss += .01 * (self.max_logvar.sum()) - .01 * (self.min_logvar.sum())
            if batch_idx == 0 and save_reconstruction:
                n = min(next_obs.size(0), 8)
                comparison = torch.cat([
                    next_obs[:n].narrow(start=0, length=self.imlength, dim=1)
                    .contiguous().view(
                        -1, self.input_channels, self.imsize, self.imsize
                    ),
                    reconstructions.view(
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

            # logger.record_tabular("test/max_logvar_mean", self.max_logvar.mean().item())
            # logger.record_tabular("test/min_logvar_mean", self.min_logvar.mean().item())
            logger.dump_tabular()
            if save_vae:
                logger.save_itr_params(epoch, self.model)  # slow...
        # logdir = logger.get_snapshot_dir()
        # filename = osp.join(logdir, 'params.pkl')
        # torch.save(self.model, filename)
        # if self.gaussian_decoder_loss:



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
        reconstructions, _, _ = self.model(data)
        img = data[0]
        recon_mse = ((reconstructions[0] - img)**2).mean().view(-1)
        img_repeated = img.expand((debug_batch_size, img.shape[0]))

        samples = ptu.randn(debug_batch_size, self.representation_size)
        random_imgs, _ = self.model.decode(samples)
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
        sample = self.model.decode(sample)[0].cpu()
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
            img_torch = ptu.from_numpy(normalize_image(img_np))
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