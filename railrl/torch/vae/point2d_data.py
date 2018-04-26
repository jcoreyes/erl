import os.path as osp
import numpy as np
from railrl.envs.multitask.point2d import MultitaskImagePoint2DEnv
import time

def get_data(N = 10000, test_p = 0.9, use_cached=True):
    filename = "/tmp/point2d_" + str(N) + ".npy"
    if use_cached and osp.isfile(filename):
        dataset = np.load(filename)
        print("loaded data from saved file", filename)
    else:
        # if not cached
        now = time.time()
        e = MultitaskImagePoint2DEnv(render_size=84, render_onscreen=False, ball_radius=1)
        dataset = np.zeros((N, 84*84))
        for i in range(N):
            dataset[i, :] = e.reset()
        print("done making training data", filename, time.time() - now)
        np.save(filename, dataset)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset

if __name__ == "__main__":
    get_data(10000)
