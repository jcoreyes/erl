# import tensorflow as tf
# import numpy as np
# import mnist_data
# import os
from rlkit.torch.vae.vae import VAE
# import plot_utils
# import glob
# import ss.path

# import argparse
from rlkit.launchers.arglauncher import run_variants

def experiment(variant):
    m = VAE()
    for epoch in range(10):
        m.train_epoch(epoch)
        m.test_epoch(epoch)
        m.dump_samples(epoch)

if __name__ == "__main__":
    run_variants(experiment, [dict()])
