# import tensorflow as tf
# import numpy as np
# import mnist_data
# import os
from railrl.torch.vae.conv_vae import ConvVAE
# import plot_utils
# import glob
# import ss.path

# import argparse
from railrl.launchers.arglauncher import run_variants

def experiment(variant):
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    m = ConvVAE(representation_size, beta=beta)
    for epoch in range(10):
        m.train_epoch(epoch)
        m.test_epoch(epoch)
        m.dump_samples(epoch)

if __name__ == "__main__":
    variants = []

    for representation_size in [2]:
        for beta in [5.0]:
            variant = dict(
                beta=beta,
                representation_size=representation_size,
            )
            variants.append(variant)
    run_variants(experiment, variants, run_id=0)
