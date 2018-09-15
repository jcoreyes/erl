import skvideo.io
import skvideo.datasets
import matplotlib.pyplot as plt
import numpy as np


def flip_up_down(video):
    video = video[:, 2:-2, 2:-2, :]
    new_video = np.zeros_like(video)
    new_video[:, :84, :, :] = video[:, 84:, :, :]
    new_video[:, 84:, :, :] = video[:, :84, :, :]
    return new_video


if __name__ == "__main__":
    videodata = skvideo.io.vread(
        # "/home/vitchyr/git/rig-blog-post/pick-and-place.mp4"
        "/home/vitchyr/git/rig-blog-post/push.mp4"
    )
    inds = [(0, 5), (2, 1), (0, 3), (1, 2), ]
    new_videodata = flip_up_down(videodata)
    skvideo.io.vwrite(
        # "/home/vitchyr/git/rig-blog-post/pick-and-place-flipped.mp4",
        "/home/vitchyr/git/rig-blog-post/push-new.mp4",
        new_videodata
    )
