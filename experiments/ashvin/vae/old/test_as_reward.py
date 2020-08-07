# import tensorflow as tf
# import numpy as np
# import mnist_data
# import os
from rlkit.torch.vae.conv_vae import ConvVAE
# import plot_utils
# import glob
# import ss.path

# import argparse
from rlkit.launchers.arglauncher import run_variants
import joblib

def experiment(variant):
    c = joblib.load("/Users/ashvin/data/s3doodad/ashvin/vae/point2d-conv/run0/id0/params.pkl")
    import pdb; pdb.set_trace()

    beta = variant["beta"]
    representation_size = variant["representation_size"]
    m = ConvVAE(representation_size, beta=beta)
    for epoch in range(10):
        m.train_epoch(epoch)
        m.test_epoch(epoch)
        m.dump_samples(epoch)

if __name__ == "__main__":
    variants = []

    for representation_size in [2, 8, 16, 32]:
        for beta in [0.5, 5.0, 50.0]:
            variant = dict(
                beta=beta,
                representation_size=representation_size,
            )
            variants.append(variant)
    run_variants(experiment, variants, run_id=4)
