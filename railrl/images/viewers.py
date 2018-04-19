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

def sawyer_init_viewer(viewer):
    viewer.cam.trackbodyid = 0
    viewer.cam.distance = 1.0

    # robot view
    #rotation_angle = 90
    #cam_dist = 1
    #cam_pos = np.array([0, 0.5, 0.2, cam_dist, -45, rotation_angle])

    # 3rd person view
    cam_dist = 0.3
    rotation_angle = 270
    cam_pos = np.array([0, 1.0, 0.5, cam_dist, -45, rotation_angle])

    # top down view
    #cam_dist = 0.2
    #rotation_angle = 0
    #cam_pos = np.array([0, 0, 1.5, cam_dist, -90, rotation_angle])

    for i in range(3):
        viewer.cam.lookat[i] = cam_pos[i]
    viewer.cam.distance = cam_pos[3]
    viewer.cam.elevation = cam_pos[4]
    viewer.cam.azimuth = cam_pos[5]
    viewer.cam.trackbodyid = -1
