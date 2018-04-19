import os.path as osp
import numpy as np
from railrl.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from railrl.envs.wrappers import ImageMujocoEnv
import time
import cv2

def get_data(N = 10000, test_p = 0.9, use_cached=True):
    filename = "/tmp/pusher2d_" + str(N) + ".npy"
    if use_cached and osp.isfile(filename):
        dataset = np.load(filename).astype(np.float32) / 255.0 # temporary hack
        print("loaded data from saved file", filename)
    else:
        # if not cached
        now = time.time()
        e = CylinderXYPusher2DEnv()
        e = ImageMujocoEnv(e, 84, "topview", transpose=True)
        dataset = np.zeros((N, 3*84*84))
        for i in range(N):
            if i % 100 == 0:
                e.reset()
            u = np.random.rand(3) * 50 - 25
            img, _, _, _ = e.step(u)
            dataset[i, :] = img
            cv2.imshow('img', img.reshape(3, 84, 84).transpose())
            cv2.waitKey(1)
        print("done making training data", filename, time.time() - now)
        np.save(filename, dataset)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset

if __name__ == "__main__":
    get_data(10000, use_cached=False)
