import torch
import torch.nn as nn
from torchvision import datasets, transforms
from rlkit.torch import pytorch_util as ptu
from os import path as osp
from sklearn import neighbors
import numpy as np
from torchvision.utils import save_image
import time
from torchvision.transforms import ColorJitter, RandomResizedCrop, Resize
from PIL import Image
from rlkit.misc.asset_loader import load_local_or_remote_file
import os
from tqdm import tqdm
import pickle
import sys

from rlkit.torch.vae.initial_state_pixelcnn import GatedPixelCNN
import rlkit.torch.vae.pixelcnn_utils
from rlkit.torch.vae.vq_vae import VQ_VAE

from rlkit.core import logger

"""
data loaders
"""

def train_pixelcnn(
    vqvae_path,
    batch_size=32,
    epochs=100,
    log_interval=100,
    num_workers=4,
    n_layers=15,
    learning_rate=3e-4,
    train_data_kwargs=None,
    test_data_kwargs=None,
    save=True,
    gen_samples=True,
    max_batches_per_iteration=-1,
):
    # Load VQVAE + Define Args
    vqvae = load_local_or_remote_file(vqvae_path)
    vqvae.to(ptu.device)
    vqvae.eval()

    root_len = vqvae.root_len
    num_embeddings = vqvae.num_embeddings
    embedding_dim = vqvae.embedding_dim
    cond_size = vqvae.num_embeddings
    imsize = vqvae.imsize
    discrete_size = root_len * root_len
    representation_size = embedding_dim * discrete_size
    input_channels = vqvae.input_channels
    imlength = imsize * imsize * input_channels

    log_dir = logger.get_snapshot_dir()

    # Define data loading info
    new_path = osp.join(log_dir, 'pixelcnn.npy')

    def prep_sample_data():
        data = np.load(new_path, allow_pickle=True).item()
        train_data = data['train']#.reshape(-1, discrete_size)
        test_data = data['test']#.reshape(-1, discrete_size)
        return train_data, test_data

    def resize_dataset(data, new_imsize=48):
        resize = Resize((new_imsize, new_imsize), interpolation=Image.NEAREST)
        data["observations"] = data["observations"].reshape(-1, 50, 84 * 84 * 3)
        num_traj, traj_len = data['observations'].shape[0], data['observations'].shape[1]
        all_data = []
        for traj_i in range(num_traj):
            traj = []
            for trans_i in range(traj_len):
                x = Image.fromarray(data['observations'][traj_i, trans_i].reshape(84, 84, 3), mode='RGB')
                x = np.array(resize(x)).reshape(1, new_imsize * new_imsize * 3)
                traj.append(x)
            traj = np.concatenate(traj, axis=0).reshape(1, traj_len, -1)
            all_data.append(traj)
        data['observations'] = np.concatenate(all_data, axis=0)



    def encode_dataset(path, max_traj=None):
        data = load_local_or_remote_file(path)
        data = data.item()
        # resize_dataset(data)

        data["observations"] = data["observations"].reshape(-1, 50, imlength)

        all_data = []

        n = data["observations"].shape[0]
        N = min(max_traj or n, n)

        # vqvae.to('cpu') # 3X faster on a GPU
        for i in tqdm(range(n)):
            obs = ptu.from_numpy(data["observations"][i] / 255.0 )
            latent = vqvae.encode(obs, cont=False).reshape(-1, 50, discrete_size)
            all_data.append(latent)

        encodings = ptu.get_numpy(torch.cat(all_data, dim=0))
        return encodings

    #### Only run to encode new data ####
    train_data = encode_dataset(**train_data_kwargs)
    test_data = encode_dataset(**test_data_kwargs)
    dataset = {'train': train_data, 'test': test_data}
    np.save(new_path, dataset)
    #### Only run to encode new data ####
    train_data, test_data = prep_sample_data()


    _, _, train_loader, test_loader, _ = \
        rlkit.torch.vae.pixelcnn_utils.load_data_and_data_loaders(new_path, 'COND_LATENT_BLOCK', batch_size)


    model = GatedPixelCNN(num_embeddings, root_len**2, n_layers, n_classes=representation_size).to(ptu.device)
    criterion = nn.CrossEntropyLoss().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    """
    train, test, and log
    """

    def train():
        train_loss = []
        for batch_idx, x in enumerate(train_loader):
            start_time = time.time()
            x_comb = x.cuda()

            cond = vqvae.discrete_to_cont(x_comb[:, vqvae.discrete_size:]).reshape(x.shape[0], -1)
            x = x_comb[:, :vqvae.discrete_size].reshape(-1, root_len, root_len)

            # Train PixelCNN with images
            logits = model(x, cond)
            logits = logits.permute(0, 2, 3, 1).contiguous()

            loss = criterion(
                logits.view(-1, num_embeddings),
                x.contiguous().view(-1)
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss.append(loss.item())

            if (batch_idx + 1) % log_interval == 0:
                print('\tIter: [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                    batch_idx * len(x), len(train_loader.dataset),
                    log_interval * batch_idx / len(train_loader),
                    np.asarray(train_loss)[-log_interval:].mean(0),
                    time.time() - start_time
                ))

            if max_batches_per_iteration == batch_idx:
                break


    def test():
        start_time = time.time()
        val_loss = []
        with torch.no_grad():
            for batch_idx, x in enumerate(test_loader):
            #x = (x[:, 0]).cuda()

                x = x.cuda()
                cond = vqvae.discrete_to_cont(x[:, vqvae.discrete_size:]).reshape(x.shape[0], -1)
                x = x[:, :vqvae.discrete_size].reshape(-1, root_len, root_len)

                logits = model(x, cond)
                logits = logits.permute(0, 2, 3, 1).contiguous()
                loss = criterion(
                    logits.view(-1, num_embeddings),
                    x.contiguous().view(-1)
                )

                val_loss.append(loss.item())

        print('Validation Completed!\tLoss: {} Time: {}'.format(
            np.asarray(val_loss).mean(0),
            time.time() - start_time
        ))
        return np.asarray(val_loss).mean(0)

    def generate_samples(epoch, test=True, batch_size=64):
        if test:
            dataset = test_data
            dtype = 'test'
        else:
            dataset = train_data
            dtype = 'train'

        rand_indices = np.random.choice(dataset.shape[0], size=(8,))
        data_points = ptu.from_numpy(dataset[rand_indices, 0]).long().cuda()

        samples = []

        for i in range(8):
            env_latent = data_points[i].reshape(1, -1)
            cond = vqvae.discrete_to_cont(env_latent).reshape(1, -1)

            samples.append(vqvae.decode(cond))

            e_indices = model.generate(shape=(root_len, root_len),
                    batch_size=7, cond=cond.repeat(7, 1)).reshape(-1, root_len**2)
            samples.append(vqvae.decode(e_indices, cont=False))

        samples = torch.cat(samples, dim=0)
        filename = osp.join(log_dir, "cond_sample_{0}_{1}.png".format(dtype, epoch))
        save_image(
            samples.data.view(batch_size, input_channels, imsize, imsize).transpose(2, 3),
            filename
        )


    # def generate_samples(epoch, batch_size=64):
    #     num_samples = 8
    #     data_points = ptu.from_numpy(all_data[np.random.choice(all_data.shape[0], size=(num_samples,))]).long().cuda()

    #     envs = data_points[:, vqvae.discrete_size:]
    #     samples = []

    #     cond = vqvae.discrete_to_cont(data_points[:, vqvae.discrete_size:]).reshape(num_samples, -1)
    #     cond.repeat(num_samples - 1, 1)
    #     e_indices = model.generate(
    #                 shape=(root_len, root_len),
    #                 batch_size=(num_samples - 1) * num_samples,
    #                 cond=cond.repeat(num_samples - 1, 1)
    #                 )
    #     cond_images = vqvae.decode(cond)
    #     vqvae.decode(e_indices.reshape(-1, root_len**2), cont=False)

    BEST_LOSS = 999
    LAST_SAVED = -1
    for epoch in range(1, epochs):
        vqvae.set_pixel_cnn(model)
        print("\nEpoch {}:".format(epoch))

        model.train()
        train()
        cur_loss = test()

        if save or cur_loss <= BEST_LOSS:
            BEST_LOSS = cur_loss
            LAST_SAVED = epoch
            print("Saving model")
            pixelcnn_path = osp.join(log_dir, 'pixelcnn.pkl')
            vqvae_path = osp.join(log_dir, 'vqvae.pkl')
            pickle.dump(model, open(pixelcnn_path, "wb"))
            pickle.dump(vqvae, open(vqvae_path, "wb"))
        else:
            print("Not saving model! Last saved: {}".format(LAST_SAVED))
        if gen_samples:
            model.eval()
            generate_samples(epoch, test=True)
            generate_samples(epoch, test=False)
