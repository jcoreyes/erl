import numpy as np

def inverted_pendulum_v2_init_viewer(viewer):
    viewer.cam.trackbodyid = 0
    viewer.cam.lookat[2] = .3
    viewer.cam.distance=1
    viewer.cam.elevation = 0

def reacher_v2_init_viewer(viewer):
    viewer.cam.distance= .7
    viewer.cam.elevation = 90
    viewer.cam.azimuth = 90

def inverted_double_pendulum_init_viewer(viewer):
    viewer.cam.elevation=1.22
    viewer.cam.distance=1.8
    viewer.cam.lookat[2] = .6
    viewer.cam.trackbodyid = 0

def pusher_2d_init_viewer(viewer):
    viewer.cam.trackbodyid = 0
    viewer.cam.distance = 4.0
    rotation_angle = 90
    cam_dist = 4
    cam_pos = np.array([0, 0, 0, cam_dist, -45, rotation_angle])
    for i in range(3):
        viewer.cam.lookat[i] = cam_pos[i]
    viewer.cam.distance = cam_pos[3]
    viewer.cam.elevation = 90
    viewer.cam.azimuth = 90
    viewer.cam.trackbodyid = -1

