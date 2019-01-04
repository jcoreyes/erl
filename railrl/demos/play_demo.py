import numpy as np
import cv2

def play_demos(path):
    data = np.load(path)

    for traj in data:
        obs = traj["observations"]

        for o in obs:
            img = o["image_observation"].reshape((84, 84, 3))
            cv2.imshow('window', img)
            cv2.waitKey(100)

if __name__ == '__main__':
    play_demos("/Users/ashvin/data/s3doodad/demos/multiobj1_demos_100.npy")
