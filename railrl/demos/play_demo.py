import numpy as np
import cv2

def play_demos(path):
    data = np.load(path, allow_pickle=True)

    for traj in data:
        obs = traj["observations"]

        for o in obs:
            img = o["image_observation"].reshape(3, 500, 300).transpose()
            cv2.imshow('window', img)
            cv2.waitKey(100)

if __name__ == '__main__':
    play_demos("demo.npy")
