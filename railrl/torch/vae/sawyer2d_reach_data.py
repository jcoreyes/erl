import os.path as osp
import numpy as np

from railrl.envs.mujoco.sawyer_gripper_env import SawyerXYZEnv, SawyerPushXYEnv, \
    SawyerXYEnv
from railrl.envs.multitask.point2d import MultitaskImagePoint2DEnv
import time

from railrl.envs.wrappers import ImageMujocoEnv, NormalizedBoxEnv
from railrl.images.camera import sawyer_init_camera
import cv2


def get_data(N = 10000, test_p = 0.9, use_cached=True):
    filename = "/tmp/sawyer_" + str(N) + ".npy"
    if use_cached and osp.isfile(filename):
        dataset = np.load(filename)
        print("loaded data from saved file", filename)
    else:
        # if not cached
        now = time.time()
        # env = SawyerXYZEnv()
        env = SawyerPushXYEnv()
        env = SawyerXYEnv()
        imsize = 32
        env = NormalizedBoxEnv(ImageMujocoEnv(
            env,
            imsize=imsize,
            keep_prev=0,
            init_camera=sawyer_init_camera,
        ))

        dataset = np.zeros((N, imsize*imsize*3))
        K = imsize * imsize
        for i in range(N):
            dataset[i, :] = env.reset()
            # raw_img = dataset[i, :].reshape(imsize, imsize, 3, order='C')
            # raw_img = env._image_observation()
            # img = np.concatenate((
            #     raw_img[::-1, :, 2:3],
            #     raw_img[::-1, :, 1:2],
            #     raw_img[::-1, :, 0:1],
            # ), axis=2)
            # cv2.imshow('obs', img)
            # cv2.waitKey(1)
        print("done making training data", filename, time.time() - now)
        np.save(filename, dataset)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset

if __name__ == "__main__":
    get_data(10000)
