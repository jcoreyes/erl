
# coding: utf-8

# In[2]:


get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib notebook')


# In[3]:


"""
Skew the dataset so that it turns into generating a uniform distribution.
"""
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict
from scipy.stats import chisquare
from torch import nn as nn
from torch.autograd import Variable
from torch.distributions import Normal
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler, RandomSampler
from torch.utils.data.dataset import Dataset


# In[4]:


from railrl.misc import visualization_util as vu
from railrl.misc.ml_util import ConstantSchedule, PiecewiseLinearSchedule


# In[6]:


def gaussian_data(batch_size):
    return np.random.randn(batch_size, 2)


def uniform_truncated_data(batch_size):
    data = np.random.uniform(low=-2, high=2, size=(batch_size, 2))
    data = np.maximum(data, -1)
    data = np.minimum(data, 1)
    return data


def uniform_gaussian_data(batch_size):
    data = np.random.randn(batch_size, 2)
    data = np.maximum(data, -1)
    data = np.minimum(data, 1)
    return data


def uniform_data(batch_size):
    return np.random.uniform(low=-2, high=2, size=(batch_size, 2))


def affine_gaussian_data(batch_size):
    return (
        np.random.randn(batch_size, 2) * np.array([1, 10]) + np.array(
        [20, 1])
    )


def flower_data(batch_size):
    z_true = np.random.uniform(0, 1, batch_size)
    r = np.power(z_true, 0.5)
    phi = 0.25 * np.pi * z_true
    x1 = r * np.cos(phi)
    x2 = r * np.sin(phi)

    # Sampling form a Gaussian
    x1 = np.random.normal(x1, 0.10 * np.power(z_true, 2), batch_size)
    x2 = np.random.normal(x2, 0.10 * np.power(z_true, 2), batch_size)

    # Bringing data in the right form
    X = np.transpose(np.reshape((x1, x2), (2, batch_size)))
    X = np.asarray(X, dtype='float32')
    return X

ut_dataset = uniform_truncated_data(1000)
u_dataset = uniform_data(100000)
empty_dataset = np.zeros((0, 2))


# ---

# Make sure that chi-squared test is reasonable for X samples and Y num bins.
# 
# Based on these experiments:
# 
# Bad:    n_bins = 1000

# In[7]:


n_bin_values = [2, 5, 10, 50, 100, 500]
n_exp = len(n_bin_values)
assert n_exp % 2 == 0
plt.figure()
for i, n_bins in enumerate(n_bin_values):
    p_values = []
    for _ in range(1000):
        uniform = uniform_data(1000)
        h, xe, ye = np.histogram2d(
            uniform[:, 0],
            uniform[:, 1],
            bins=n_bins,
            range=np.array([[-2, 2], [-2, 2]]),
        )
        counts = h.flatten()
        p_values.append(chisquare(counts).pvalue)
    plt.subplot(2, n_exp // 2, i+1)
    plt.hist(p_values, bins=20)
    plt.title("# bins = {}".format(n_bins))
plt.suptitle("P values Histograms")
plt.show()


# -------------------------------------

# Instead of IPDB, use
# ```
# from IPython.core.debugger import Tracer; Tracer()()
# ```
# for debugging

# In[8]:


def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )


# In[9]:


def np_to_var(np_array, **kwargs):
    return Variable(torch.from_numpy(np_array).float(), **kwargs)


def kl_to_prior(means, log_stds, stds):
    """
    KL between a Gaussian and a standard Gaussian.

    https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    """
    return (
            - log_stds
            - 0.5  # d = 1
            + stds ** 2
            + means ** 2
    ).sum(dim=1, keepdim=True)


def log_prob(batch, decoder, latents):
    decoder_output = decoder(latents)
    half_idx = decoder_output.shape[1]//2
    decoder_means = decoder_output[:, :half_idx]
    decoder_log_stds = decoder_output[:, half_idx:]
    distribution = Normal(decoder_means, torch.ones_like(decoder_means))#, decoder_log_stds.exp())
    return distribution.log_prob(batch).sum(dim=1, keepdim=True)


class Encoder(nn.Sequential):
    def encode(self, x):
        return self.get_encoding_and_suff_stats(x)[0]

    def get_encoding_and_suff_stats(self, x):
        output = self(x)
        z_dim = output.shape[1] // 2
        means, log_stds = (
            output[:, :z_dim], output[:, z_dim:]
        )
        stds = log_stds.exp()
        epsilon = Variable(torch.randn(*means.size()))
        latents = epsilon * stds + means
        return latents, means, log_stds, stds


class Decoder(nn.Sequential):
    def decode(self, latents):
        output = self(latents)
        x_dim = output.shape[1] // 2
        means, log_stds = output[:, :x_dim], output[:, x_dim:]
        distribution = Normal(means, torch.ones_like(means))#log_stds.exp())
        return distribution.sample()


# In[10]:


def compute_train_weights(data, encoder, decoder, config):
    """
    :param data: PyTorch
    :param encoder:
    :param decoder:
    :return: PyTorch
    """
    alpha = config.get('alpha', 0)
    mode = config.get('mode', 'none')
    n_average = config.get('n_average', 3)
    orig_data_length = len(data)
    if alpha == 0 or mode == 'none':
        return np.ones(orig_data_length)
    data = np.vstack([
        data for _ in range(n_average)
    ])
    data = np_to_var(data)
    z_dim = decoder._modules['0'].weight.shape[1]
    """
    Actually compute the weights
    """
    if mode == 'biased_encoder':
        latents, means, log_stds, stds = encoder.get_encoding_and_suff_stats(
            data
        )
        importance_weights = 1
    elif mode == 'prior':
        latents = Variable(torch.randn(len(data), z_dim))
        importance_weights = 1
    elif mode == 'importance_sampling':
        latents, means, log_stds, stds = encoder.get_encoding_and_suff_stats(
            data
        )
        
        prior = Normal(0, 1)
        prior_prob = prior.log_prob(latents).sum(dim=1).exp()
        
        encoder_distrib = Normal(means, stds)
        encoder_prob = encoder_distrib.log_prob(latents).sum(dim=1).exp()
        
        importance_weights = prior_prob / encoder_prob
    else:
        raise NotImplementedError()
    
    data_prob = log_prob(data, decoder, latents).squeeze(1).exp()
    weights = importance_weights * 1. / data_prob
    weights = weights**alpha
        
    """
    Average over `n_average`
    """
        
    samples_of_results = torch.split(weights, orig_data_length, dim=0)
    # pre_avg.shape = ORIG_LEN x N_AVERAGE
    pre_avg = torch.cat(
        [x.unsqueeze(1) for x in samples_of_results],
        1,
    )
    # final.shape = ORIG_LEN
    final = torch.mean(pre_avg, dim=1, keepdim=False)
    return final.data.numpy()
    


# In[11]:


class IndexedData(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return index, self.dataset[index]


# In[12]:


def generate_vae_samples_np(decoder, n_samples):
    z_dim = decoder._modules['0'].weight.shape[1]
    generated_samples = decoder.decode(
        Variable(torch.randn(n_samples, z_dim))
    )
    return generated_samples.data.numpy()

def project_samples_np(samples):
    samples = np.maximum(samples, -1)
    samples = np.minimum(samples, 1)
    return samples


# In[13]:


def train(
    train_data,
    encoder=None,
    decoder=None,
    bs=32,
    n_samples_to_add_per_epoch=1000,
    n_epochs=100,
    skew_config=None,
    weight_loss=False,
    skew_sampling=False,
    beta_schedule=None,
    z_dim=1,
    hidden_size=32,
    save_period=10,
):
    if encoder is None:
        encoder = Encoder(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, z_dim*2),
        )
    if decoder is None:
        decoder = Decoder(
            nn.Linear(z_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4),
        )
    if beta_schedule is None:
        beta_schedule = ConstantSchedule(1)
    if skew_config is None:
        skew_config = dict(
            use_log_prob=False,
            alpha=0,
        )

    encoder_opt = Adam(encoder.parameters())
    decoder_opt = Adam(decoder.parameters())

    epochs = []
    losses = []
    kls = []
    log_probs = []
    encoders = []
    decoders = []
    train_datas = []
    weights = []
    for epoch in log_progress(range(n_epochs)):
        epoch_stats = defaultdict(list)
        if n_samples_to_add_per_epoch > 0:
            vae_samples = generate_vae_samples_np(decoder, n_samples_to_add_per_epoch)
            projected_samples = project_samples_np(vae_samples)
            train_data = np.concatenate((train_data, projected_samples), 0)
        indexed_train_data = IndexedData(train_data)
        
        all_weights = compute_train_weights(train_data, encoder, decoder, skew_config)
        all_weights_pt = np_to_var(all_weights, requires_grad=False)
        if sum(all_weights) == 0:
            all_weights[:] = 1

        if skew_sampling:
            base_sampler = WeightedRandomSampler(all_weights, len(all_weights))
        else:
            base_sampler = RandomSampler(indexed_train_data)
        
        train_dataloader = DataLoader(
            indexed_train_data,
            sampler=BatchSampler(
                base_sampler,
                batch_size=bs,
                drop_last=False,
            ),
        )
        if epoch == 0 or (epoch+1) % save_period == 0:
            epochs.append(epoch)
            encoders.append(copy.deepcopy(encoder))
            decoders.append(copy.deepcopy(decoder))
            train_datas.append(train_data)
            weights.append(all_weights)
        for i, indexed_batch in enumerate(train_dataloader):
            idxs, batch = indexed_batch

            batch = Variable(batch[0].float())

            latents, means, log_stds, stds = encoder.get_encoding_and_suff_stats(
                batch
            )
            beta = float(beta_schedule.get_value(epoch))
            kl = kl_to_prior(means, log_stds, stds)
            reconstruction_log_prob = log_prob(batch, decoder, latents)

            elbo = - kl*beta + reconstruction_log_prob
            if weight_loss:               
                idxs = torch.cat(idxs)
                weights = all_weights_pt[idxs].unsqueeze(1)
                weighted_elbo = elbo * weights
                loss = -(weights * elbo).mean()
            else:
                loss = - elbo.mean()
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            loss.backward()
            encoder_opt.step()
            decoder_opt.step()

            epoch_stats['losses'].append(loss.data.numpy())
            epoch_stats['kls'].append(kl.mean().data.numpy())
            epoch_stats['log_probs'].append(reconstruction_log_prob.mean().data.numpy())
        losses.append(np.mean(epoch_stats['losses']))
        kls.append(np.mean(epoch_stats['kls']))
        log_probs.append(np.mean(epoch_stats['log_probs']))
    
    return epochs, encoders, decoders, train_datas, losses, kls, log_probs


# In[14]:


def show_heatmap(train_results, skew_config, xlim=(-5, 5), ylim=(-5, 5), resolution=20):
    encoder, decoder, losses, kls, log_probs = train_results
    
    def get_prob_batch(batch):
        return 1./compute_train_weights(batch, encoder, decoder, skew_config)
    
    heat_map = vu.make_heat_map(get_prob_batch, xlim, ylim, resolution=resolution, batch=True)
    plt.figure()
    vu.plot_heatmap(heat_map)
    plt.show()


# In[15]:


def visualize_results(results, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), n_vis=1000):
    for epoch, encoder, decoder, vis_samples_np in zip(*results[:4]):
        plt.figure()
        plt.suptitle("Epoch {}".format(epoch))
        
        n_samples = len(vis_samples_np)
        skip_factor = max(n_samples // n_vis, 1)
        vis_samples_np = vis_samples_np[::skip_factor]
        
        vis_samples = np_to_var(vis_samples_np)
        latents = encoder.encode(vis_samples)
        z_dim = latents.shape[1]
        reconstructed_samples = decoder.decode(latents).data.numpy()
        generated_samples = decoder.decode(
            Variable(torch.randn(n_vis, z_dim))
        ).data.numpy()
        projected_generated_samples = project_samples_np(generated_samples)

        plt.subplot(2, 2, 1)
        plt.plot(generated_samples[:, 0], generated_samples[:, 1], '.')
        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.title("Generated Samples")
        
        
        plt.subplot(2, 2, 2)
        plt.plot(projected_generated_samples[:, 0], projected_generated_samples[:, 1], '.')
        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.title("Projected Generated Samples")
        
        
        plt.subplot(2, 2, 3)
        plt.plot(reconstructed_samples[:, 0], reconstructed_samples[:, 1], '.')
        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.title("Reconstruction")

        
        plt.subplot(2, 2, 4)
        plt.plot(vis_samples_np[:, 0], vis_samples_np[:, 1], '.')
        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.title("Original Samples")
    plt.show()


# In[16]:


def plot_uniformness(results, n_samples=10000, n_bins=5):
    generated_frac_on_border_lst = []
    dataset_frac_on_border_lst = []
    p_values = []  # computed using chi squared test
    for epoch, encoder, decoder, vis_samples_np in zip(*results[:4]):

        z_dim = decoder._modules['0'].weight.shape[1]
        generated_samples = decoder.decode(
            Variable(torch.randn(n_samples, z_dim))
        ).data.numpy()
        projected_generated_samples = project_samples_np(generated_samples)

        orig_n_samples_on_border = np.mean(
            np.any(vis_samples_np == 1, axis=1)
            + np.any(vis_samples_np == -1, axis=1)
        )
        dataset_frac_on_border_lst.append(orig_n_samples_on_border)
        gen_n_samples_on_border = np.mean(
            np.any(projected_generated_samples == 1, axis=1)
            + np.any(projected_generated_samples == -1, axis=1)
        )
        generated_frac_on_border_lst.append(gen_n_samples_on_border)
        
        
        # Is this data sampled from a uniform distribution? Compute p-value
        h, xe, ye = np.histogram2d(
            projected_generated_samples[:, 0],
            projected_generated_samples[:, 1],
            bins=n_bins,
            range=np.array([[-1, 1], [-1, 1]]),
        )
        counts = h.flatten()
        p_values.append(chisquare(counts).pvalue)
    
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(np.array(generated_frac_on_border_lst))
    plt.xlabel('epoch')
    plt.ylabel('Fraction of points along border')
    plt.title("Sampled from VAE")
    
    plt.subplot(1, 3, 2)
    plt.plot(np.array(dataset_frac_on_border_lst))
    plt.xlabel('epoch')
    plt.ylabel('Fraction of points along border')
    plt.title("All Aggregated Samples")
  
    plt.subplot(1, 3, 3)
    plt.plot(np.array(p_values))
    plt.xlabel('epoch')
    plt.ylabel('Uniform Distribution Goodness of fit: p-value')
    plt.title("Sampled from VAE")
    plt.show()


# In[17]:


def plot_curves(train_results):
    *_, losses, kls, log_probs = train_results
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(np.array(losses))
    plt.title("Training Loss")
    plt.subplot(1, 3, 2)
    plt.plot(np.array(kls))
    plt.title("KLs")
    plt.subplot(1, 3, 3)
    plt.plot(np.array(log_probs))
    plt.title("Log Probs")
    plt.xlabel("Epoch")
    plt.show()


# In[18]:


plt.close('all')


# # Model Uniform Distribution
# This is a sanity check to make sure this VAE architecture/training procedure is good enough to model a 2D uniform distribution, given a 2D Gaussian prior.

# In[ ]:


uniform_results = train(
    u_dataset,
    bs=32,
    n_epochs=100,
    n_samples_to_add_per_epoch=0,
    skew_config=dict(
        alpha=0,
        mode='none',
        n_average=2,
    ),
    skew_sampling=False,
    z_dim=16,
    hidden_size=32,
    save_period=5,
)


# In[ ]:


plot_uniformness(uniform_results)
plot_curves(uniform_results)
visualize_results(uniform_results, xlim=(-3, 3), ylim=(-3, 3))


# # Skew online dataset to make uniform distribution

# In[ ]:


all_results = {}
for mode in [
    'importance_sampling',
    'biased_encoder',
    'prior'
]:
    all_results[mode] = train(
        empty_dataset,
        bs=32,
        n_epochs=10000,
        n_samples_to_add_per_epoch=10,
        skew_config=dict(
            alpha=1,
            mode=mode,
            n_average=100,
        ),
        skew_sampling=True,
        z_dim=16,
        hidden_size=32,
        save_period=100,
    )


# In[234]:


plot_uniformness(all_results['importance_sampling'], n_samples=10000, n_bins=5)
plot_curves(all_results['importance_sampling'])
visualize_results(all_results['importance_sampling'], xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), n_vis=1000)


# In[235]:


plot_uniformness(all_results['biased_encoder'], n_samples=10000, n_bins=5)
plot_curves(all_results['biased_encoder'])
visualize_results(all_results['biased_encoder'], xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), n_vis=1000)


# In[236]:


plot_uniformness(all_results['prior'], n_samples=10000, n_bins=5)
plot_curves(all_results['prior'])
visualize_results(all_results['prior'], xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), n_vis=1000)

