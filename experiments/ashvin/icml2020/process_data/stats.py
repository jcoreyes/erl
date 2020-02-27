"""Load and print summary statistics of the given trajectories"""

import pickle
import numpy as np
import argparse

def print_stats(args):
    # file = "/home/ashvin/data/s3doodad/demos/icml2020/hand/pen_bc1.npy"
    # file = "/home/ashvin/data/s3doodad/demos/icml2020/hand/pen_bc2.npy"
    # file = "/home/ashvin/data/s3doodad/demos/icml2020/hand/pen.npy"
    # data = pickle.load(open(file, "rb"))

    data = np.load(open(args.file, "rb"), allow_pickle=True)

    returns = []
    path_lengths = []

    print("num trajectories", len(data))

    for path in data:
        rewards = path["rewards"]
        returns.append(np.sum(rewards))
        path_lengths.append(len(rewards))

    print("returns")
    print("min", np.min(returns))
    print("max", np.max(returns))
    print("mean", np.mean(returns))
    print("std", np.std(returns))

    print("path lengths")
    print("min", np.min(path_lengths))
    print("max", np.max(path_lengths))
    print("mean", np.mean(path_lengths))
    print("std", np.std(path_lengths))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    args = parser.parse_args()

    print_stats(args)
