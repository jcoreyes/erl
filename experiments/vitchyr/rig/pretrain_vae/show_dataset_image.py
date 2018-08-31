import argparse
import uuid

import cv2
import numpy as np

filename = str(uuid.uuid4())


def vis(args):
    imgs = np.load(args.file)
    for image_obs in imgs:
        im = image_obs.reshape(3, 48, 48).transpose()
        cv2.imshow('img', im)
        cv2.waitKey(100)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()

    vis(args)
