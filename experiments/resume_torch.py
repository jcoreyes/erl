"""
Fine tune a trained policy/qf
"""
import argparse

import joblib

import torch

import railrl.torch.pytorch_util as ptu

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str,
                        help='Path to snapshot file to fine tune.')
    args = parser.parse_args()

    ptu.set_gpu_mode(True)

    data = torch.load(args.path, "cuda")
    algo = data['algorithm']
    # algo.to("cpu")
    # algo.to("cuda")
    algo.train()
