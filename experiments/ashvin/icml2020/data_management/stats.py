"""Load and print summary statistics of the given trajectories"""

import pickle
import numpy as np
import argparse

def print_stat(array, name):
    print(name)
    print("min", np.min(array))
    print("max", np.max(array))
    print("mean", np.mean(array))
    print("std", np.std(array))

def print_stats(args):
    # file = "/home/ashvin/data/s3doodad/demos/icml2020/hand/pen_bc1.npy"
    # file = "/home/ashvin/data/s3doodad/demos/icml2020/hand/pen_bc2.npy"
    # file = "/home/ashvin/data/s3doodad/demos/icml2020/hand/pen.npy"
    # data = pickle.load(open(file, "rb"))

    data = np.load(open(args.file, "rb"), allow_pickle=True)

    returns = []
    path_lengths = []
    terminals = []

    print("num trajectories", len(data))

    for path in data:
        rewards = path["rewards"]
        actions = path["actions"]
        print_stat(actions, "actions")
        returns.append(np.sum(rewards))
        path_lengths.append(len(rewards))
        terminals.append(np.sum(path["terminals"]))

    print_stat(returns, "returns")
    print_stat(path_lengths, "path_lengths")
    print_stat(terminals, "terminals")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    args = parser.parse_args()

    print_stats(args)
