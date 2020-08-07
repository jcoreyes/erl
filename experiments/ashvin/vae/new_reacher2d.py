# import tensorflow as tf
# import numpy as np
# import mnist_data
# import os
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.vae.conv_vae import ConvVAE
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from rlkit.torch.vae.reacher2d_data import get_data
# import plot_utils
# import glob
# import ss.path

# import argparse
from rlkit.launchers.arglauncher import run_variants
import rlkit.torch.pytorch_util as ptu


def experiment(variant):
    if variant["use_gpu"]:
        ptu.set_gpu_mode(True)
        ptu.set_device(0)

    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data = get_data(10000)
    m = ConvVAE(representation_size, input_channels=3)
    t = ConvVAETrainer(train_data, test_data, m, beta=beta)
    for epoch in range(1000):
        t.train_epoch(epoch)
        t.test_epoch(epoch)
        t.dump_samples(epoch)

if __name__ == "__main__":
    variants = []
    representation_sizes = [8, 16]
    for representation_size in representation_sizes:
        for beta in [0]:
            variant = dict(
                beta=beta,
                representation_size=representation_size,
                use_gpu='True',
            )
            variants.append(variant)
    for i, variant in enumerate(variants):
        n_seeds = 1
        exp_prefix = 'reacher_autoencoder_train_'+str(representation_sizes[i])
        mode = 'local'
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                snapshot_mode='gap',
                snapshot_gap=20,
                exp_prefix=exp_prefix,
                variant=variant,
                use_gpu=True,
            )
