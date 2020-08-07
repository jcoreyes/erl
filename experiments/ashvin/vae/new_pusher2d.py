# import tensorflow as tf
# import numpy as np
# import mnist_data
# import os
from rlkit.torch.vae.conv_vae import ConvVAE
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from rlkit.torch.vae.pusher2d_data import get_data
# import plot_utils
# import glob
# import ss.path

# import argparse
from rlkit.launchers.arglauncher import run_variants
import rlkit.torch.pytorch_util as ptu

def experiment(variant):
    if variant["use_gpu"]:
        gpu_id = variant["gpu_id"]
        ptu.set_gpu_mode(True)
        ptu.set_device(gpu_id)

    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data = get_data(10000)
    m = ConvVAE(representation_size, input_channels=3)
    t = ConvVAETrainer(train_data, test_data, m, beta=beta)
    for epoch in range(1001):
        t.train_epoch(epoch)
        t.test_epoch(epoch)
        t.dump_samples(epoch)

if __name__ == "__main__":
    variants = []
    for representation_size in [4, 8, 16, 32]:
        for beta in [5.0]:
            variant = dict(
                beta=beta,
                representation_size=representation_size,
                snapshot_mode="gap",
                snapshot_gap=100,
            )
            variants.append(variant)
    run_variants(experiment, variants, run_id=9)

