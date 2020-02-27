import numpy as np

x = np.load("demos2.npy")

import skvideo.io

for j in range(len(x)):
    d = x[j]
    obs_right = []
    obs_left = []
    for i in range(len(d)):
        obs_left.append(d[i].left_shoulder_rgb)
        obs_right.append(d[i].right_shoulder_rgb)

    videodata = (np.array(obs_left) * 255).astype(int)
    filename = "demo_left_%d.mp4" % j
    skvideo.io.vwrite(filename, videodata)

    videodata = (np.array(obs_right) * 255).astype(int)
    filename = "demo_right_%d.mp4" % j
    skvideo.io.vwrite(filename, videodata)
