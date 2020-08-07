# import tensorflow as tf
# import numpy as np
# import mnist_data
# import os
from rlkit.torch.vae.conv_vae import ConvVAE
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
    m = ConvVAE(train_data, test_data, representation_size, beta=beta, use_cuda=True, input_channels=3)
    for epoch in range(50):
        m.train_epoch(epoch)
        m.test_epoch(epoch)
        m.dump_samples(epoch)

if __name__ == "__main__":
    variants = []
    train_data, test_data = get_data(100)
    import ipdb; ipdb.set_trace()
    for representation_size in [2, 4, 8, 16]:
        for beta in [0.5, 5.0, 50]:
            variant = dict(
                beta=beta,
                representation_size=representation_size,
            )
            variants.append(variant)
    run_variants(experiment, variants, run_id=0)
