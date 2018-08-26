import skvideo.io
import skvideo.datasets
import matplotlib.pyplot as plt
import numpy as np

def to_long(videodata, inds, x_padding=4, y_padding=4, bgr=False, flip=False):
    Y = 2*(84+y_padding)
    X = 84+x_padding
    new_videodata = np.zeros((101 * len(inds), Y, X, 3), dtype=np.uint8)
    kk = 0
    for i, j in inds:
        k = i * 6 + j
        start = 101 * kk
        end = 101 * (kk+1)
        y_start = Y * i
        y_end = Y * (i+1)
        x_start = X * j
        x_end = X * (j+1)
        vdata = videodata[:, y_start:y_end, x_start:x_end, :]
        if bgr:
            vdata = rgb(vdata)
        if flip:
            vdata = np.flip(vdata, 1)
        new_videodata[start:end, :, :, :] = vdata
        kk += 1
    return new_videodata

def rgb(data):
    return np.roll(data, 1, axis=-1)

if __name__ == "__main__": # how to use this:
    videodata = skvideo.io.vread("/Users/ashvin/Downloads/video_2550_env.mp4")
    inds = [(0, 5), (2, 1), (0, 3), (1, 2), ]
    new_videodata = to_long(videodata, inds, x_padding=0, y_padding=0, bgr=False, flip=True)
    skvideo.io.vwrite("/Users/ashvin/data/s3doodad/media/videos/videos/pickplace.mp4", new_videodata)
