import argparse
import uuid

import cv2
import numpy as np
import joblib

from railrl.data_management.images import normalize_image
from railrl.torch.core import eval_np

filename = str(uuid.uuid4())

NUM_SHOWN = 3


def vis(args):
    imgs = np.load(args.ds)
    vae = joblib.load(args.file)
    losses = []
    for i, image_obs in enumerate(imgs):
        img = normalize_image(image_obs)
        recon, *_ = eval_np(vae, img)
        error = ((recon - img)**2).sum()
        losses.append((i, error))

    losses.sort(key=lambda x: -x[1])

    for rank, (i, error) in enumerate(losses[:NUM_SHOWN]):
        image_obs = imgs[i]
        recon, *_ = eval_np(vae, normalize_image(image_obs))

        img = image_obs.reshape(3, 48, 48).transpose()
        rimg = recon.reshape(3, 48, 48).transpose()

        cv2.imshow(
            "image, rank {}, loss {}".format(rank, error),
            img
        )
        cv2.imshow(
            "recon, rank {}, loss {}".format(rank, error),
            rimg
        )
        print("rank {}\terror {}".format(rank, error))
    for j, (i, error) in enumerate(losses[-NUM_SHOWN:]):
        rank = len(losses) - j - 1
        image_obs = imgs[i]
        recon, *_ = eval_np(vae, normalize_image(image_obs))

        img = image_obs.reshape(3, 48, 48).transpose()
        rimg = recon.reshape(3, 48, 48).transpose()

        cv2.imshow(
            "image, rank {}, loss {}".format(rank, error),
            img
        )
        cv2.imshow(
            "recon, rank {}, loss {}".format(rank, error),
            rimg
        )
        print("rank {}\terror {}".format(rank, error))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    parser.add_argument('ds', type=str, help='dataset path')
    args = parser.parse_args()

    vis(args)
