import argparse
import uuid

import cv2
import numpy as np

filename = str(uuid.uuid4())


def vis(args):
    presampled_goals_np = np.load(args.file)
    presampled_goals = dict(presampled_goals_np.flatten()[0])
    imgs = presampled_goals['image_desired_goal']
    for image_obs in imgs:
        if image_obs.size == 6912:
            im = image_obs.reshape(3, 48, 48).transpose()
        else:
            im = image_obs.reshape(3, 84, 84).transpose()
        im = im[:, :, ::-1]
        cv2.imshow('img', im)
        cv2.waitKey(10)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()

    vis(args)
