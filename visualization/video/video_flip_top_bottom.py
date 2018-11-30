import skvideo.io
import skvideo.datasets
import numpy as np


def flip(video_data):
    new_videodata = np.zeros_like(video_data)
    new_videodata[:, :84, :, :] = video_data[:, 84:, :, :]
    new_videodata[:, 84:, :, :] = video_data[:, :84, :, :]
    return new_videodata


def rgb(data):
    return np.roll(data, 1, axis=-1)


if __name__ == "__main__":  # how to use this:
    videodata = skvideo.io.vread(
        "/home/vitchyr/Videos/research/rig-blog-post/door.gif")
    new_videodata = flip(videodata)
    skvideo.io.vwrite(
        "/home/vitchyr/Videos/research/rig-blog-post/door_flipped.gif",
        new_videodata)
